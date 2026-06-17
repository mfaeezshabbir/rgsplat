// splat.wgsl — WGSL compute shader for Gaussian Splatting
// Runs on GPU via wgpu. Two passes:
//   Pass 1 (project):    Gaussian3D[] → SplatInfo2D[] + depth keys
//   Pass 2 (rasterize):  sorted SplatInfo2D[] → RGBA output texture

// ─────────────────────────────────────────────────────────────────────────────
// Shared types (must match Rust #[repr(C)] layouts exactly)
// ─────────────────────────────────────────────────────────────────────────────

struct Gaussian3D {
    position:       vec3<f32>,
    opacity_logit:  f32,
    rotation:       vec4<f32>,  // quaternion [w, x, y, z]
    log_scale:      vec3<f32>,
    _pad:           f32,
    sh_coeffs:      array<f32, 48>,
}

struct SplatInfo2D {
    sx:      f32,   // screen x
    sy:      f32,   // screen y
    conic_a: f32,   // Σ⁻¹[0,0]
    conic_b: f32,   // Σ⁻¹[0,1]
    conic_c: f32,   // Σ⁻¹[1,1]
    color_r: f32,
    color_g: f32,
    color_b: f32,
    alpha:   f32,
    depth:   f32,
}

struct CameraUniforms {
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    cam_pos:    vec3<f32>,
    width:      u32,
    height:     u32,
    fx:         f32,
    fy:         f32,
    cx:         f32,
    cy:         f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       gaussians:  array<Gaussian3D>;
@group(0) @binding(1) var<storage, read_write> splats_out: array<SplatInfo2D>;
@group(0) @binding(2) var<uniform>             camera:     CameraUniforms;
@group(0) @binding(3) var<storage, read_write> depth_keys: array<u32>;  // for radix sort

// ─────────────────────────────────────────────────────────────────────────────
// Math utilities
// ─────────────────────────────────────────────────────────────────────────────

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    // q = [w, x, y, z]
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    return mat3x3<f32>(
        vec3(1.0 - 2.0*(y*y + z*z),       2.0*(x*y + w*z),       2.0*(x*z - w*y)),
        vec3(      2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z),       2.0*(y*z + w*x)),
        vec3(      2.0*(x*z + w*y),       2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y)),
    );
}

fn build_covariance_3d(rotation: vec4<f32>, log_scale: vec3<f32>) -> mat3x3<f32> {
    let R = quat_to_mat3(rotation);
    let s = exp(log_scale);
    let S = mat3x3<f32>(
        vec3(s.x, 0.0, 0.0),
        vec3(0.0, s.y, 0.0),
        vec3(0.0, 0.0, s.z),
    );
    let RS = R * S;
    return RS * transpose(RS);  // Σ = R S Sᵀ Rᵀ
}

fn project_covariance_2d(
    sigma3d: mat3x3<f32>,
    p_cam:   vec3<f32>,
    fx:      f32,
    fy:      f32,
) -> mat3x3<f32> {
    let tz = p_cam.z;
    let tx = p_cam.x;
    let ty = p_cam.y;

    // Jacobian of perspective projection
    let J = mat3x3<f32>(
        vec3(fx / tz, 0.0,  -fx * tx / (tz * tz)),
        vec3(0.0, fy / tz,  -fy * ty / (tz * tz)),
        vec3(0.0, 0.0, 0.0),
    );

    // Extract view rotation (upper-left 3×3 of view matrix)
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );

    let T = J * W;
    var sigma2d = T * sigma3d * transpose(T);
    // Low-pass anti-aliasing filter
    sigma2d[0][0] += 0.3;
    sigma2d[1][1] += 0.3;
    return sigma2d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: Project Gaussians (one thread per Gaussian)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) { return; }

    let g = gaussians[idx];

    // Transform to camera space
    let p_world = vec4<f32>(g.position, 1.0);
    let p_cam4  = camera.view * p_world;
    let p_cam   = p_cam4.xyz;
    let tz = p_cam.z;

    // Behind camera or too close
    if tz <= 0.001 {
        splats_out[idx].alpha = 0.0;
        depth_keys[idx] = 0xFFFFFFFFu;  // sort to end
        return;
    }

    let alpha = sigmoid(g.opacity_logit);
    if alpha < (1.0 / 255.0) {
        splats_out[idx].alpha = 0.0;
        depth_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    // Screen-space center
    let sx = camera.fx * p_cam.x / tz + camera.cx;
    let sy = camera.fy * p_cam.y / tz + camera.cy;

    // Project covariance
    let sigma3d = build_covariance_3d(g.rotation, g.log_scale);
    let sigma2d = project_covariance_2d(sigma3d, p_cam, camera.fx, camera.fy);

    // Invert 2×2 covariance → conic
    let a = sigma2d[0][0];
    let b = sigma2d[0][1];
    let c = sigma2d[1][1];
    let det = a * c - b * b;
    if abs(det) < 1e-8 {
        splats_out[idx].alpha = 0.0;
        depth_keys[idx] = 0xFFFFFFFFu;
        return;
    }
    let inv_det = 1.0 / det;

    // DC color (SH degree 0)
    let SH_C0 = 0.28209479177387814f;
    let color = vec3<f32>(
        clamp(0.5 + g.sh_coeffs[ 0] * SH_C0, 0.0, 1.0),
        clamp(0.5 + g.sh_coeffs[16] * SH_C0, 0.0, 1.0),
        clamp(0.5 + g.sh_coeffs[32] * SH_C0, 0.0, 1.0),
    );

    // Write output
    splats_out[idx].sx      = sx;
    splats_out[idx].sy      = sy;
    splats_out[idx].conic_a = c * inv_det;
    splats_out[idx].conic_b = -b * inv_det;
    splats_out[idx].conic_c = a * inv_det;
    splats_out[idx].color_r = color.r;
    splats_out[idx].color_g = color.g;
    splats_out[idx].color_b = color.b;
    splats_out[idx].alpha   = alpha;
    splats_out[idx].depth   = tz;

    // Encode depth as sortable u32 (float bit pattern, back-to-front)
    let depth_f = tz;
    let depth_bits = bitcast<u32>(depth_f);
    // Flip for back-to-front (largest float first after sort)
    depth_keys[idx] = ~depth_bits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: Tile rasterization (one workgroup per tile)
// Each 16×16 tile → 256 threads (one per pixel)
// ─────────────────────────────────────────────────────────────────────────────

@group(1) @binding(0) var<storage, read>       sorted_splats: array<SplatInfo2D>;
@group(1) @binding(1) var<storage, read>       tile_ranges:   array<vec2<u32>>;  // (start, end) per tile
@group(1) @binding(2) var                      output_tex:    texture_storage_2d<rgba8unorm, write>;

const TILE = 16u;

@compute @workgroup_size(16, 16)
fn rasterize(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let tile_idx = wid.y * ((camera.width + TILE - 1u) / TILE) + wid.x;
    let range    = tile_ranges[tile_idx];
    let px       = wid.x * TILE + lid.x;
    let py       = wid.y * TILE + lid.y;

    if px >= camera.width || py >= camera.height { return; }

    var color = vec3<f32>(0.0);
    var T     = 1.0;   // transmittance

    for (var i = range.x; i < range.y; i++) {
        let s = sorted_splats[i];
        if s.alpha < (1.0 / 255.0) { continue; }

        let dx = f32(px) + 0.5 - s.sx;
        let dy = f32(py) + 0.5 - s.sy;

        // Evaluate Gaussian: exp(-½ [dx,dy] Σ⁻¹ [dx,dy]ᵀ)
        let power = -0.5 * (s.conic_a * dx * dx
                          + 2.0 * s.conic_b * dx * dy
                          + s.conic_c * dy * dy);
        if power > 0.0 { continue; }

        let alpha = min(s.alpha * exp(power), 0.99);
        if alpha < (1.0 / 255.0) { continue; }

        // Front-to-back compositing
        color += vec3(s.color_r, s.color_g, s.color_b) * alpha * T;
        T *= 1.0 - alpha;

        if T < 0.0001 { break; }  // ray saturated
    }

    // Linear → sRGB approximation (pow(x, 1/2.2))
    let out_color = vec4<f32>(pow(color, vec3(1.0 / 2.2)), 1.0 - T);
    textureStore(output_tex, vec2<i32>(i32(px), i32(py)), out_color);
}
