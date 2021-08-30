# wgsl-rs

A linker for WGSL (WebGPU Shading Language) shaders. This crate aims to provide Rust-like module support and straightforward compilation for WGSL shaders. In other words, no more thousand-line shader files full of duplicated code. 

Useful WebGPU resources:
- [Official spec](https://gpuweb.github.io/gpuweb/)
- [Rust implementation](https://github.com/gfx-rs/wgpu) 
- [In-depth tutorial](https://sotrh.github.io/learn-wgpu)

### Usage

---
<!--
## Intro to GPUs
The next few sections form a short introduction to modern gpu programming standards/architecture/hardware. It is targeted towards graphics programmers who are new to `wgpu` but have used other graphics APIs before. However, it is also written to be comprehendible by the average non-graphics programmer, so feel free to skip ahead if this is below you. The wgpu-specific section starts [here](#what-is-wgpu).

#### What is GPU programming?
GPU/graphics programming is a unique & far-reaching area of software. The term began as a general reference to all the processes involved in getting computers to display pixels on a screen. Nowadays, as the performant architecture of GPUs has become desirable in other fields, the term "gpu programming" can be more abstractly redefined as everything involved with writing programs which take advantage of both central and graphics processors.

#### Why do we need GPUs?
Most modern CPUs contain 4-8 cores. Cores are like workers capable of executing tasks. More cores allows for more tasks to be executed simultaneously. By design, the number of cores in a GPU far eclipses that of your average CPU. This makes it ideal for situations in which a larger task can be subdivided into many smaller tasks which _do not depend on each other_, and can thus be run in parallel.

#### Why do we still need CPUs if GPUs have more cores?
Two general reasons:
- Most tasks cannot be broken up into hundreds of independent subtasks, because the subtasks end up forming a complex, nonuniform temporal dependency graph.
- CPU cores and GPU cores are not the same; CPUs are optimized to do many things well, while GPUs are designed as a hardware acceleration for a specific _type_ of problem.

#### What is a graphics API?
Graphics APIs facilitate running operations on the GPU from the CPU. Unless you are a hardware engineer at Nvidia, graphics APIs are the lowest-level interfaces to the gpu that you can use. Major features they provide include:

- Generation of gpu-executable machine code, usually via some shading language like GLSL, HLSL, or WGSL.
- Control over gpu memory allocation, mutation, and liberation from the cpu, usually by requiring the user to maintain two descriptions of their data (one for cpu and one for gpu).
- Control over the gpu pipeline structure (via defining the synchronous series of asynchronous shaders, buffer visibilities/formats, data flow, etc).

Together, these solutions allow graphics APIs to act as the main interface for programmers interested in writing parallel tasks (also known as _shaders_) which run on the GPU. The following is a description of the major ones in use today.

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
