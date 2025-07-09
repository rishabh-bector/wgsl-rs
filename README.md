> Note: this project is in an incomplete and abandoned state.

# RSL: Rust Shader Language

RSL is a shader language designed as an alternative to WGSL. It is intended for use with [wgpu-rs](https://github.com/gfx-rs/wgpu), an excellent implementation of the [WebGPU](https://gpuweb.github.io/gpuweb/) graphics API in Rust. The `wgpu` crate has become the standard for graphics developers using Rust. RSL is built on top of WGPU; it's intended to make WGSL more fun. If it's more efficient/reliable, that was an accident. But you never know...

Features:
- [Shader linking](#shader-linking): separate your shaders into organized modules and directories via `pub` and `use` syntax.
- [Data Consolidation](#data-consolidation): generate vertex attribute structs, uniform structs and vertex buffer layouts in Rust, at compile time, from your shader code.
- [Opinionated Syntax](#opinionated-syntax): RSL features a more complete Rust-like syntax than WGSL. See the linked section for the reasoning behind RSL.
- [Boilerplate](#boilerplate): Add boilerplate generation features as added.

Possible other features of RSL:
- enums in WGSL
- impls in WGSL
- no need for functions to be defined in a certain order

Useful WebGPU resources:
- [Official spec](https://gpuweb.github.io/gpuweb/)
- [In-depth tutorial](https://sotrh.github.io/learn-wgpu)


## Overview
---
### Shader Linking

`wgsl-rs` allows you to separate your shaders into multiple files, and compose modules via Rust-like `pub` and `use` declarations. In other words, no more thousand-line shader file/modules full of duplicated code. 

#### Life before `wgsl-rs`

Typically, one writes their shader module in one `.wgsl` file, which might look like this:

```rust
[[block]]
struct Camera {
    view_proj: mat4x4<f32>;
};

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] uvs: vec2<f32>;
};

struct VertexOutput {
    [[location(0)]] uvs: vec2<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

struct Bruh {
    hello: vec2<f32>;
    lmao: i32;
};

// Vertex shader

[[group(0), binding(0)]]
var<uniform> camera: Camera;

[[stage(vertex)]]
fn main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.uvs = model.uvs;
    out.position = camera.view_proj * vec4<f32>(model.position, 1.0);

    var x: Bruh = Bruh(vec2<f32>(1.0, 2.0), 1);

    return out;
}

// Fragment shader

[[group(1), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(1), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uvs);
}
```

The shader module is then compiled from its source using a `wgpu::Device`:

```rust
let module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
    label: Some("example_shader"),
    flags: wgpu::ShaderFlags::all(),
    source: wgpu::ShaderSource::Wgsl(include_str!("example.wgsl").into()),
});
```
#### The `wgsl-rs` Experience

For formatting/highlighting purposes, a new file type is used: `.rsl` (rust shader language). All your shader files live under one `rsl/` directory anywhere in your crate. The following directory structure is enforced:

```
rsl/
|  modules/
|  |  <module 1>/
|  |  | vert.rsl
|  |  | frag.rsl
|  |  <module 2>/
|  |  | vert.rsl
|  |  | frag.rsl
|  common/
|  |  <common file 1>.rsl 
|  |  <common file 2>.rsl 
```

Modules consist of a vert/frag shader pair, which can be compiled into a WGSL module; other types of shaders are not yet supported. Modules can `use` from common files, and vertex/fragment stages within a module can `use` from each other. There are no "common modules" or common directories because shader modules should not be too large. Common files can `use` from each other as long as cyclical dependencies are avoided.

# WRITE A FULL SYNTAX TUTORIAL, THIS IS NOT ENOUGH

The example shader given above is tiny, but it's enough to demonstrate RSL.  `example.wgsl` as RSL.., and moving the object uniforms into a common file. The file tree would look like this:

```
rsl/
|  modules/
|  |  example/
|  |  | vert.rsl
|  |  | frag.rsl
|  common/
|  |  object.rsl 
```

And the files:

`vert.rsl`
```rust
use camera::CameraData;

#[vertex]
struct VertexInput {
    position: Vec3,
    uvs: Vec2,
}

#[transport]
pub struct VertexOutput {
    #[position]: Vec4,
    uvs: Vec2,
}

fn main(
    vert: VertexInput,
    #[uniform] camera: CameraData,
) -> VertexOutput {
    let out: VertexOutput::new(
        camera.view_proj * vec4(vert.position, 1.0),
        vert.uvs,
    );
        
    return out;
}
```

`frag.rsl`
```rust
use vert::VertexOutput;

[[group(1), binding(0)]]
var diffuse_t: texture_2d<f32>;
[[group(1), binding(1)]]
var diffuse_s: sampler;

[[stage(fragment)]]
fn main(frag: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(diffuse_t, diffuse_s, frag.uvs);
}
```

`object.rsl`
```rust
pub struct ObjectUniforms {
    model_matrix: mat4x4<f32>;
};
```

Now, you can generate the same `wgpu::ShaderModule` as the original example via a macro in your Rust code:

`main.rs`
```rust
// in this case, rsl/ is in src/

#[shaders(src/rsl)]
pub enum Shaders {
    Example,
}

fn main() {
    // .. create window, wgpu surface/adapter/device, etc.

    let example_module = shader_module!(Shader::Example);
}
```

An RSL shader tree can have as many modules as you want.


---

### Data Consolidation

#### Life before `wgsl-rs`

Currently, all data which is to be buffered from cpu to gpu must be described by developers multiple times. The first is in the shader itself. Above, both `ObjectUniforms` and `VertexInput` are gpu input data structs (one represents per-vertex data, while the other is a uniform). After defining them in the shader, you have to redefine them in Rust, which would look something like this:

```rust
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniforms {
    model_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct VertexInput {
    position: [f32; 3],
    uvs: [f32; 2],
}
```

Uniform structs derive from `bytemuck` so that we can write them to a byte buffer each frame. Vertex data is only buffered on start. You then have to describe the vertex attribute structure a _third_ time, to use in your render pipeline:


```rust
let vertex_input_layout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::InputStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x3,
        }
    ]
};
```

#### The `wgsl-rs` Experience

To improve this, `wgsl-rs` can eliminate _all_ structure redefinition by generating vertex attribute structs, uniform structs, and vertex buffer layouts in Rust, directly from your shader code, at compile time. Generating structs for uniforms and vertex inputs is as simple as decorating their shader definitions with the `#[uniform]` or `#[vertex]` macros. 

In `object.rsl` from the original example:
```rust
#[uniform]
pub struct ObjectUniforms {
    model_matrix: mat4x4<f32>;
};
```

And for the vertex input in `vert.rsl`:
```rust
#[vertex]
struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] uvs: vec2<f32>;
};
```

---

### Opinionated Syntax

RSL features a more complete Rust-like syntax than WGSL. Some may prefer the traditional GLSL experience, and others WGSL. I argue that RSL is preferrable to both, because:
- It has features that they do not, such as enums and pub/use.
- Strict adherance to Rust syntax is preferrable to a mix of Rust and GLSL.

<!--
## Intro to GPUs
The next few sections form a short introduction to modern gpu programming standards/architecture/hardware. It is targeted towards graphics programmers who are new to `wgpu` but have used other graphics APIs before. However, it is also written to be comprehendible by the average non-graphics programmer, so feel free to skip ahead if this is below you. The wgpu-specific section starts [here](#what-is-wgpu).

#### What is GPU programming?
GPU/graphics programming is a unique & far-reaching area of software. The term began as a general reference to all the processes involved in getting computers to display pixels on a screen. Nowadays, as the performant architecture of GPUs has become desirable in other fields, the term "gpu programming" refers to everything involved with writing programs which take advantage of both the central and graphics processors.

#### Why do we need GPUs?
Most modern CPUs contain 4-8 cores. Cores are like workers capable of executing tasks. More cores allows for more tasks to be executed simultaneously. By design, the number of cores in a GPU far eclipses that of your average CPU. This makes it ideal for situations in which a larger task can be subdivided into many smaller tasks which do not depend on each other and can thus be run in parallel.

#### Why do we still need CPUs if GPUs have more cores?
Two general reasons:
- Most tasks cannot be broken up into hundreds of independent subtasks, because the subtasks end up forming a complex graph of dependencies.
- CPU cores and GPU cores are not the same; CPUs are optimized to do many things well, while GPUs are designed as a hardware acceleration for a specific type of problem.

#### What is a graphics API?
Graphics APIs help developers run operations on the GPU from the CPU. Unless you're a hardware engineer at Nvidia, a graphics API is likely the lowest-level interface to the gpu that you have access to. Major features include:

- Generation of gpu-executable machine code, usually via some shading language like GLSL, HLSL, or WGSL.
- Control over gpu memory allocation, mutation, and liberation from the cpu, usually by requiring the user to maintain two descriptions of their data (one for cpu and one for gpu).
- Control over the gpu pipeline structure (via defining processing stages, memory scope and format, and other configuration).

Together, these features allow a graphics API to act as the main interface for programmers interested in writing parallel tasks (also known as _shaders_) which run on the GPU. The following is a description of the major/standard APIs in use today.

| Name | Description |
| --- | --- |
| OpenGL | - First major cross-platform hardware-acceleration API, developed in 1991 <br> - Still very popular, with bindings for all relevant languages <br> - Best option for beginners, as [high quality learning resources exist](https://learnopengl.com/) <br> - Supported by all major hardware vendors: Intel, Nvidia, AMD, Google/Android, Apple, and Qualcomm <br><br> OpenGL is a higher-level API compared to Vulkan or Metal, which makes it significantly easier to learn. This also means that it cannot fully take advantage of modern GPU features or offer the same performance as those APIs. |
| DirectX | - Developed by Microsoft in 1995 <br> - Was OpenGL's main competitor for a long time <br> - Only relevant for Windows machines <br><br> DirectX can be thought of as a Windows-optimized analog to OpenGL. |
| Metal | - Developed by Apple in 2014 <br> - Only relevant for Apple machines <br><br> Metal is like Apple's version of DirectX, except a lot more modern. Following the release of Metal, Apple decided to deprecate their OpenGL support.  |
| Vulkan | - Developed in 2015 by the Kronos Group, who also maintain OpenGL. <br> - Cross-platform, although the Apple implementation currently uses Metal under the hood. <br> - Lowest-level API compared to everything that came before it. <br> - More performant than OpenGL/DirectX/Metal, in part due to improved CPU/GPU usage balancing and multithreading-friendly design. <br><br> Vulkan is supposed to be the ultimate modern graphics API. |

---

#### What is WebGPU?
Browsers have become one of the most desirable application distribution platforms, being more convenient and secure than traditional apps which have to be downloaded & installed. As a result, several graphics APIs with the goal of allowing graphical web applications to use the host's GPU have been thought up. Currently, though, only one is being widely used: `WebGL`, which exposes OpenGL to the browser.

WebGPU is the successor to WebGL, and uses Vulkan instead of OpenGL, for the most part. Technically, it is designed as a platform-agnostic lossless abstraction over Vulkan, Metal, and DirectX. This has allowed WGPU to become a complete Vulkan-level API which is compatible with web browsers, but not tied to them. The result is that there aren't many reasons to choose a graphics API other than WGPU _even_ when writing offline applications, because it gives you the level of Vulkan while also being compatible with DirectX and OpenGL.


#### What is WGSL?

-->
