//! GL (EGL) rendering backend. Mirrors picom's `src/backend/gl/egl.c`.
//!
//! - [`first_light`]: headless pbuffer smoke test (validates EGL + glow + GL).
//! - [`GlBackend`]: an EGL **window** surface on the composite overlay, with a
//!   textured-quad blit program and [`GlBackend::present_window_pixmap`] —
//!   bind an X window's pixmap as a GL texture (EGLImage) and draw it. This is
//!   the heart of compositing; the renderer drives it over the window stack.

use std::ffi::c_void;

use anyhow::{anyhow, bail, Result};
use glow::HasContext as _;
use khronos_egl as egl;

/// `glEGLImageTargetTexture2DOES(target, image)` — loaded via eglGetProcAddress.
type ImageTargetTexture2DOes = unsafe extern "system" fn(target: u32, image: *const c_void);

/// `EGL_NATIVE_PIXMAP_KHR` (from EGL_KHR_image_pixmap; not exported by khronos-egl).
const EGL_NATIVE_PIXMAP_KHR: egl::Enum = 0x30B0;

const BLIT_VS: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos;   // unit quad, 0..1
uniform vec4 u_rect;                   // x, y, w, h  (pixels, top-left origin)
uniform vec2 u_screen;                 // screen w, h
out vec2 v_tex;
void main() {
    v_tex = a_pos;                                  // (0,0) = window top-left
    vec2 px = u_rect.xy + a_pos * u_rect.zw;        // pixel position
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0); // flip Y into GL NDC
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"#;

const BLIT_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform sampler2D u_tex;
uniform float u_opacity;              // whole-window opacity, 0..1
out vec4 frag;
// Premultiplied-alpha output: paired with glBlendFunc(ONE, ONE_MINUS_SRC_ALPHA)
// over an opaque clear this yields  dst = rgb*a + dst*(1-a)  — straight "over".
void main() { frag = vec4(texture(u_tex, v_tex).rgb * u_opacity, u_opacity); }
"#;

/// Soft drop-shadow fragment shader (reuses [`BLIT_VS`], so it also has `u_rect`
/// = the shadow quad and `u_screen`). Casts only to the **left** and **bottom**
/// of the window (light from the top-right), fading off over the radius with a
/// rounded bottom-left corner. Output is premultiplied black for the same
/// ONE/ONE_MINUS_SRC_ALPHA blend as the blit.
const SHADOW_FS: &str = r#"#version 330 core
in vec2 v_tex;
uniform vec4 u_rect;     // shadow quad: x, y, w, h  (px, top-left origin)
uniform vec4 u_inner;    // caster window rect: x, y, w, h  (px)
uniform vec2 u_shadow;   // x = radius (falloff px), y = strength (max alpha)
out vec4 frag;
void main() {
    vec2 p  = u_rect.xy + v_tex * u_rect.zw;   // fragment pixel position
    vec2 lo = u_inner.xy;
    vec2 hi = u_inner.xy + u_inner.zw;
    float r = u_shadow.x;
    float dist = 1e9;
    // Distance to the left edge segment, shortened at the top by r, cast only
    // to the left — so the band rounds off before the top-left corner.
    if (p.x <= lo.x) {
        float cy = clamp(p.y, lo.y + r, hi.y);
        dist = min(dist, length(vec2(lo.x - p.x, p.y - cy)));
    }
    // Distance to the bottom edge segment, shortened at the right by r, cast
    // only below — so the band rounds off before the bottom-right corner.
    if (p.y >= hi.y) {
        float cx = clamp(p.x, lo.x, hi.x - r);
        dist = min(dist, length(vec2(p.x - cx, p.y - hi.y)));
    }
    float a = u_shadow.y * (1.0 - smoothstep(0.0, r, dist));
    frag = vec4(0.0, 0.0, 0.0, a);
}
"#;

/// Drop-shadow parameters (constants for now; config-driven on the roadmap).
const SHADOW_RADIUS: f32 = 12.0; // falloff distance to the left/bottom (px)
const SHADOW_STRENGTH: f32 = 0.45; // peak shadow alpha

/// One window to composite: its named pixmap, on-screen rect (top-left origin,
/// pixels, border included), whole-window opacity (`0.0..=1.0`), and whether to
/// draw a drop shadow behind it.
#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub pixmap: u32,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub opacity: f32,
    pub shadow: bool,
}

#[derive(Debug, Clone)]
pub struct GlInfo {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
}

fn load_glow(egl: &egl::DynamicInstance<egl::EGL1_5>) -> glow::Context {
    unsafe {
        glow::Context::from_loader_function(|name| match egl.get_proc_address(name) {
            Some(p) => p as *const c_void,
            None => std::ptr::null(),
        })
    }
}

/// Compile + link a vertex/fragment program. Requires a current GL context
/// (caller invokes this from within the backend's context).
fn make_program(gl: &glow::Context, vs: &str, fs: &str) -> Result<glow::NativeProgram> {
    unsafe {
        let program = gl.create_program().map_err(|e| anyhow!("create_program: {e}"))?;
        let mut shaders = Vec::new();
        for (ty, src) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
            let sh = gl.create_shader(ty).map_err(|e| anyhow!("create_shader: {e}"))?;
            gl.shader_source(sh, src);
            gl.compile_shader(sh);
            if !gl.get_shader_compile_status(sh) {
                bail!("shader compile failed: {}", gl.get_shader_info_log(sh));
            }
            gl.attach_shader(program, sh);
            shaders.push(sh);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            bail!("program link failed: {}", gl.get_program_info_log(program));
        }
        for sh in shaders {
            gl.detach_shader(program, sh);
            gl.delete_shader(sh);
        }
        Ok(program)
    }
}

/// Headless EGL/GL smoke test (pbuffer + clear + readback of GL info).
pub fn first_light() -> Result<GlInfo> {
    let xlib = x11_dl::xlib::Xlib::open().map_err(|e| anyhow!("dlopen libX11: {e}"))?;
    // Must precede the first Xlib call: lets libX11 install its locks so Mesa's
    // driver threads don't trip the "Xlib is not thread-safe" stderr warning.
    if unsafe { (xlib.XInitThreads)() } == 0 {
        bail!("XInitThreads failed");
    }
    let xdisplay = unsafe { (xlib.XOpenDisplay)(std::ptr::null()) };
    if xdisplay.is_null() {
        bail!("XOpenDisplay(NULL) failed (is DISPLAY set?)");
    }
    let egl = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required() }
        .map_err(|e| anyhow!("load libEGL.so.1 (>=1.5): {e}"))?;
    let display = unsafe { egl.get_display(xdisplay as egl::NativeDisplayType) }
        .ok_or_else(|| anyhow!("eglGetDisplay returned no display"))?;
    let (major, minor) = egl.initialize(display).map_err(|e| anyhow!("eglInitialize: {e:?}"))?;
    tracing::info!("EGL {major}.{minor} initialised");
    egl.bind_api(egl::OPENGL_API).map_err(|e| anyhow!("eglBindAPI: {e:?}"))?;

    let config_attribs = [
        egl::SURFACE_TYPE, egl::PBUFFER_BIT,
        egl::RENDERABLE_TYPE, egl::OPENGL_BIT,
        egl::RED_SIZE, 8, egl::GREEN_SIZE, 8, egl::BLUE_SIZE, 8, egl::ALPHA_SIZE, 8,
        egl::NONE,
    ];
    let config = egl
        .choose_first_config(display, &config_attribs)
        .map_err(|e| anyhow!("eglChooseConfig: {e:?}"))?
        .ok_or_else(|| anyhow!("no matching EGLConfig"))?;
    let surface = egl
        .create_pbuffer_surface(display, config, &[egl::WIDTH, 64, egl::HEIGHT, 64, egl::NONE])
        .map_err(|e| anyhow!("eglCreatePbufferSurface: {e:?}"))?;
    let context = egl
        .create_context(display, config, None, &[egl::NONE])
        .map_err(|e| anyhow!("eglCreateContext: {e:?}"))?;
    egl.make_current(display, Some(surface), Some(surface), Some(context))
        .map_err(|e| anyhow!("eglMakeCurrent: {e:?}"))?;

    let gl = load_glow(&egl);
    let info = unsafe {
        let info = GlInfo {
            vendor: gl.get_parameter_string(glow::VENDOR),
            renderer: gl.get_parameter_string(glow::RENDERER),
            version: gl.get_parameter_string(glow::VERSION),
        };
        gl.clear_color(0.10, 0.40, 0.80, 1.0);
        gl.clear(glow::COLOR_BUFFER_BIT);
        gl.finish();
        let err = gl.get_error();
        if err != glow::NO_ERROR {
            bail!("GL error after clear: 0x{err:04x}");
        }
        info
    };
    tracing::info!(vendor=%info.vendor, renderer=%info.renderer, version=%info.version, "GL first light OK");

    let _ = egl.make_current(display, None, None, None);
    let _ = egl.destroy_context(display, context);
    let _ = egl.destroy_surface(display, surface);
    let _ = egl.terminate(display);
    unsafe { (xlib.XCloseDisplay)(xdisplay) };
    Ok(info)
}

/// An EGL/GL context rendering into a target X window (the composite overlay).
pub struct GlBackend {
    xlib: x11_dl::xlib::Xlib,
    xdisplay: *mut x11_dl::xlib::Display,
    egl: egl::DynamicInstance<egl::EGL1_5>,
    display: egl::Display,
    surface: egl::Surface,
    context: egl::Context,
    gl: glow::Context,
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    u_rect: Option<glow::NativeUniformLocation>,
    u_screen: Option<glow::NativeUniformLocation>,
    u_tex: Option<glow::NativeUniformLocation>,
    u_opacity: Option<glow::NativeUniformLocation>,
    // Drop-shadow program (shares the unit-quad VAO and BLIT_VS).
    shadow_program: glow::NativeProgram,
    s_rect: Option<glow::NativeUniformLocation>,
    s_screen: Option<glow::NativeUniformLocation>,
    s_inner: Option<glow::NativeUniformLocation>,
    s_shadow: Option<glow::NativeUniformLocation>,
    image_target: ImageTargetTexture2DOes,
}

impl GlBackend {
    /// Create an EGL window surface on X window `window` (X visual `visual_id`),
    /// a current GL context with vsync, and the blit program + quad.
    pub fn new(window: u32, visual_id: u32) -> Result<Self> {
        let xlib = x11_dl::xlib::Xlib::open().map_err(|e| anyhow!("dlopen libX11: {e}"))?;
        // Must precede the first Xlib call: lets libX11 install its locks so Mesa's
        // driver threads don't trip the "Xlib is not thread-safe" stderr warning.
        if unsafe { (xlib.XInitThreads)() } == 0 {
            bail!("XInitThreads failed");
        }
        let xdisplay = unsafe { (xlib.XOpenDisplay)(std::ptr::null()) };
        if xdisplay.is_null() {
            bail!("XOpenDisplay(NULL) failed");
        }
        let egl = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required() }
            .map_err(|e| anyhow!("load libEGL.so.1 (>=1.5): {e}"))?;
        let display = unsafe { egl.get_display(xdisplay as egl::NativeDisplayType) }
            .ok_or_else(|| anyhow!("eglGetDisplay returned no display"))?;
        let (major, minor) = egl.initialize(display).map_err(|e| anyhow!("eglInitialize: {e:?}"))?;
        tracing::info!("EGL {major}.{minor} initialised (window backend)");
        egl.bind_api(egl::OPENGL_API).map_err(|e| anyhow!("eglBindAPI: {e:?}"))?;

        let attribs = [
            egl::SURFACE_TYPE, egl::WINDOW_BIT,
            egl::RENDERABLE_TYPE, egl::OPENGL_BIT,
            egl::RED_SIZE, 8, egl::GREEN_SIZE, 8, egl::BLUE_SIZE, 8,
            egl::NONE,
        ];
        let mut configs: Vec<egl::Config> = Vec::with_capacity(64);
        egl.choose_config(display, &attribs, &mut configs)
            .map_err(|e| anyhow!("eglChooseConfig: {e:?}"))?;
        let config = configs
            .into_iter()
            .find(|c| {
                egl.get_config_attrib(display, *c, egl::NATIVE_VISUAL_ID).ok()
                    == Some(visual_id as egl::Int)
            })
            .ok_or_else(|| anyhow!("no EGLConfig matching overlay visual 0x{visual_id:x}"))?;

        let surface = unsafe {
            egl.create_window_surface(display, config, (window as usize) as egl::NativeWindowType, None)
        }
        .map_err(|e| anyhow!("eglCreateWindowSurface: {e:?}"))?;
        let context = egl
            .create_context(display, config, None, &[egl::NONE])
            .map_err(|e| anyhow!("eglCreateContext: {e:?}"))?;
        egl.make_current(display, Some(surface), Some(surface), Some(context))
            .map_err(|e| anyhow!("eglMakeCurrent: {e:?}"))?;
        let _ = egl.swap_interval(display, 1); // vsync

        let gl = load_glow(&egl);
        let renderer = unsafe { gl.get_parameter_string(glow::RENDERER) };

        // glEGLImageTargetTexture2DOES (texture-from-EGLImage).
        let image_target: ImageTargetTexture2DOes = {
            let f = egl
                .get_proc_address("glEGLImageTargetTexture2DOES")
                .ok_or_else(|| anyhow!("glEGLImageTargetTexture2DOES unavailable"))?;
            unsafe { std::mem::transmute::<_, ImageTargetTexture2DOes>(f) }
        };

        // Blit program + unit-quad VAO.
        let (program, vao, u_rect, u_screen, u_tex, u_opacity) = unsafe {
            let program = make_program(&gl, BLIT_VS, BLIT_FS)?;
            let vao = gl.create_vertex_array().map_err(|e| anyhow!("vao: {e}"))?;
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().map_err(|e| anyhow!("vbo: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let bytes = std::slice::from_raw_parts(verts.as_ptr() as *const u8, 32);
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
            gl.bind_vertex_array(None);
            (
                program,
                vao,
                gl.get_uniform_location(program, "u_rect"),
                gl.get_uniform_location(program, "u_screen"),
                gl.get_uniform_location(program, "u_tex"),
                gl.get_uniform_location(program, "u_opacity"),
            )
        };

        // Shadow program: same vertex shader (unit quad -> u_rect), shadow FS.
        let (shadow_program, s_rect, s_screen, s_inner, s_shadow) = unsafe {
            let sp = make_program(&gl, BLIT_VS, SHADOW_FS)?;
            (
                sp,
                gl.get_uniform_location(sp, "u_rect"),
                gl.get_uniform_location(sp, "u_screen"),
                gl.get_uniform_location(sp, "u_inner"),
                gl.get_uniform_location(sp, "u_shadow"),
            )
        };
        tracing::info!(%renderer, window, "GL window backend ready (blit + shadow programs loaded)");

        Ok(GlBackend {
            xlib, xdisplay, egl, display, surface, context, gl,
            program, vao, u_rect, u_screen, u_tex, u_opacity,
            shadow_program, s_rect, s_screen, s_inner, s_shadow, image_target,
        })
    }

    /// Clear the surface to a colour and present.
    pub fn clear_present(&self, r: f32, g: f32, b: f32, a: f32) -> Result<()> {
        unsafe {
            self.gl.clear_color(r, g, b, a);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.egl
            .swap_buffers(self.display, self.surface)
            .map_err(|e| anyhow!("eglSwapBuffers: {e:?}"))?;
        Ok(())
    }

    /// Bind an X window's pixmap as a GL texture and blit it at `(x,y,w,h)` over
    /// a cleared overlay, then present. (Single-window path; the renderer will
    /// loop this over the stack without clearing between windows.)
    #[allow(clippy::too_many_arguments)]
    pub fn present_window_pixmap(
        &self,
        pixmap: u32,
        x: i32, y: i32, w: i32, h: i32,
        screen_w: i32, screen_h: i32,
    ) -> Result<()> {
        let buffer = unsafe { egl::ClientBuffer::from_ptr((pixmap as usize) as egl::EGLClientBuffer) };
        let no_ctx = unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) };
        let attribs = [egl::IMAGE_PRESERVED as egl::Attrib, 1, egl::ATTRIB_NONE];
        let image = self
            .egl
            .create_image(self.display, no_ctx, EGL_NATIVE_PIXMAP_KHR, buffer, &attribs)
            .map_err(|e| anyhow!("eglCreateImage(pixmap 0x{pixmap:x}): {e:?}"))?;

        let (e_bind, e_draw) = unsafe {
            let tex = self.gl.create_texture().map_err(|e| anyhow!("create_texture: {e}"))?;
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            (self.image_target)(glow::TEXTURE_2D, image.as_ptr() as *const c_void);
            let e_bind = self.gl.get_error();

            self.gl.viewport(0, 0, screen_w, screen_h);
            self.gl.clear_color(0.05, 0.05, 0.07, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);

            self.gl.use_program(Some(self.program));
            self.gl.uniform_4_f32(self.u_rect.as_ref(), x as f32, y as f32, w as f32, h as f32);
            self.gl.uniform_2_f32(self.u_screen.as_ref(), screen_w as f32, screen_h as f32);
            self.gl.uniform_1_i32(self.u_tex.as_ref(), 0);
            self.gl.uniform_1_f32(self.u_opacity.as_ref(), 1.0);
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            self.gl.bind_vertex_array(None);
            let e_draw = self.gl.get_error();

            self.gl.delete_texture(tex);
            (e_bind, e_draw)
        };
        if e_bind != glow::NO_ERROR || e_draw != glow::NO_ERROR {
            tracing::warn!("GL errors: after image-bind=0x{e_bind:04x}, after draw=0x{e_draw:04x}");
        } else {
            tracing::info!("blit GL ok (no errors)");
        }
        self.egl
            .swap_buffers(self.display, self.surface)
            .map_err(|e| anyhow!("eglSwapBuffers: {e:?}"))?;
        let _ = self.egl.destroy_image(self.display, image);
        Ok(())
    }

    /// Composite a stack of [`Quad`]s: clear once, blit each bottom-to-top with
    /// its opacity, present once. Items that fail to bind are skipped.
    pub fn present_windows(&self, items: &[Quad], screen_w: i32, screen_h: i32) -> Result<()> {
        tracing::trace!(items = items.len(), screen_w, screen_h, "present");
        unsafe {
            self.gl.viewport(0, 0, screen_w, screen_h);
            self.gl.clear_color(0.05, 0.05, 0.07, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            // Premultiplied-alpha "over" so per-window opacity (and the black
            // shadows) blend onto the clear and the windows already drawn beneath.
            self.gl.enable(glow::BLEND);
            self.gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.bind_vertex_array(Some(self.vao));
            // Shadow program's per-frame constants.
            self.gl.use_program(Some(self.shadow_program));
            self.gl.uniform_2_f32(self.s_screen.as_ref(), screen_w as f32, screen_h as f32);
            self.gl.uniform_2_f32(self.s_shadow.as_ref(), SHADOW_RADIUS, SHADOW_STRENGTH);
            // Blit program's per-frame constants.
            self.gl.use_program(Some(self.program));
            self.gl.uniform_2_f32(self.u_screen.as_ref(), screen_w as f32, screen_h as f32);
            self.gl.uniform_1_i32(self.u_tex.as_ref(), 0);
        }
        for &Quad { pixmap, x, y, w, h, opacity, shadow } in items {
            // Drop shadow first, so the window is drawn on top of its own shadow
            // and each window's shadow is cast over whatever is already beneath it.
            if shadow {
                let (fx, fy, fw, fh) = (x as f32, y as f32, w as f32, h as f32);
                unsafe {
                    self.gl.use_program(Some(self.shadow_program));
                    // Quad = bounding box of the left+bottom L: extend left and down by the radius.
                    self.gl.uniform_4_f32(
                        self.s_rect.as_ref(),
                        fx - SHADOW_RADIUS,
                        fy,
                        fw + SHADOW_RADIUS,
                        fh + SHADOW_RADIUS,
                    );
                    self.gl.uniform_4_f32(self.s_inner.as_ref(), fx, fy, fw, fh);
                    self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                }
            }
            let buffer =
                unsafe { egl::ClientBuffer::from_ptr((pixmap as usize) as egl::EGLClientBuffer) };
            let no_ctx = unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) };
            let attribs = [egl::IMAGE_PRESERVED as egl::Attrib, 1, egl::ATTRIB_NONE];
            let image = match self
                .egl
                .create_image(self.display, no_ctx, EGL_NATIVE_PIXMAP_KHR, buffer, &attribs)
            {
                Ok(i) => i,
                Err(e) => {
                    tracing::warn!("create_image(pixmap 0x{pixmap:x}) failed: {e:?}");
                    continue;
                }
            };
            unsafe {
                let tex = match self.gl.create_texture() {
                    Ok(t) => t,
                    Err(_) => {
                        let _ = self.egl.destroy_image(self.display, image);
                        continue;
                    }
                };
                self.gl.use_program(Some(self.program)); // back to blit (shadow may have switched it)
                self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                self.gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                (self.image_target)(glow::TEXTURE_2D, image.as_ptr() as *const c_void);
                self.gl.uniform_4_f32(self.u_rect.as_ref(), x as f32, y as f32, w as f32, h as f32);
                self.gl.uniform_1_f32(self.u_opacity.as_ref(), opacity);
                self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                self.gl.delete_texture(tex);
            }
            let _ = self.egl.destroy_image(self.display, image);
        }
        unsafe {
            self.gl.bind_vertex_array(None);
        }
        self.egl
            .swap_buffers(self.display, self.surface)
            .map_err(|e| anyhow!("eglSwapBuffers: {e:?}"))?;
        Ok(())
    }
}

impl Drop for GlBackend {
    fn drop(&mut self) {
        let _ = self.egl.make_current(self.display, None, None, None);
        let _ = self.egl.destroy_surface(self.display, self.surface);
        let _ = self.egl.destroy_context(self.display, self.context);
        let _ = self.egl.terminate(self.display);
        unsafe { (self.xlib.XCloseDisplay)(self.xdisplay) };
    }
}
