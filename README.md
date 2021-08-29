# wgsl-rs

A linker for WGSL (WebGPU Shading Language) shaders. This crate aims to provide Rust-like module support and straightforward compilation for WGSL shaders. In other words, no more thousand-line shader files full of duplicated code. 

Useful WebGPU resources:
- [Official spec](https://gpuweb.github.io/gpuweb/)
- [Rust implementation](https://github.com/gfx-rs/wgpu) 
- [In-depth tutorial](https://sotrh.github.io/learn-wgpu)

---

The following is a very short intro to modern gpu programming standards/architecture/hardware. It is targeted towards graphics programmers who are new to WGPU but have used other graphics APIs before. However, it is also written to be comprehendible by the average non-graphics programmer.

##### What is gpu programming?
GPU/graphics programming is a wide and unique area of software. The term began as a general reference to all the processes involved in getting computers to display pixels on a screen. Nowadays, as the concurrent hardware acceleration of GPUs has become desirable in other fields, the term GPU programming can be defined more abstractly as the processes involved in writing programs which take advantage of both CPU and GPU architectures.

##### What is a graphics API?
Graphics APIs facilitate running operations on the graphics processing unit (GPU) from the CPU. Unless you are a hardware engineer at Nvidia, a graphics API is the lowest-level interface to the GPU you will ever need.

##### What is WGPU?
WebGPU is a low-level graphics API originally designed as a platform-agnostic lossless abstraction over several existing APIs.

##### What is WGSL?

### Usage
> Note: If you are new to WebGPU, I 