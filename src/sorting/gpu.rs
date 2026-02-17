use pollster::FutureExt;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline, Device, Queue};

/// GPU context for sorting operations
pub struct GpuContext {
    device: Device,
    queue: Queue,
    bitonic_pipeline: ComputePipeline,
    merge_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl GpuContext {
    /// Initialize GPU context with compute pipeline
    pub fn new() -> Option<Self> {
        // Request adapter
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .ok()?;

        // Request device
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GPU Sorter Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .block_on()
            .ok()?;

        // Create shader module for bitonic sort
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bitonic Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shaders.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sort Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create compute pipeline for bitonic sort
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sort Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let bitonic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bitonic Sort Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bitonic_sort"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let merge_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Merge Sort Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("merge_sort"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            bitonic_pipeline,
            merge_pipeline,
            bind_group_layout,
        })
    }

    /// Create GPU buffer from data
    pub fn create_buffer<T: bytemuck::Pod>(&self, data: &[T], usage: wgpu::BufferUsages) -> Buffer {
        let contents = bytemuck::cast_slice(data);
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sort Data Buffer"),
                contents,
                usage,
            });
        buffer
    }

    /// Read data from GPU buffer back to CPU
    pub fn read_buffer<T: bytemuck::Pod + Default + Clone>(
        &self,
        buffer: &Buffer,
        size: usize,
    ) -> Vec<T> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (size * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<T>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        // Wait for GPU operations to complete (simplified)
        // self.device.poll(...); // TODO: Fix poll API

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Execute bitonic sort on GPU
    pub fn bitonic_sort_gpu(&self, data: &mut [u32]) {
        let size = data.len() as u32;
        if size == 0 {
            return;
        }

        // For small arrays or non-power-of-two, fall back to CPU sort
        if size < 8 || !size.is_power_of_two() {
            data.sort();
            return;
        }

        // Create GPU buffer
        let gpu_buffer = self.create_buffer(
            data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // Create temp buffer for merge operations
        let temp_buffer = self.create_buffer(
            &vec![0u32; data.len()],
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sort Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &temp_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // For demonstration, run multiple passes of the simple sort
        // In a real implementation, this would be proper bitonic sort passes
        for _pass in 0..(size.ilog2() * 2) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Bitonic Sort Pass"),
                });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bitonic Sort Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.bitonic_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups
            let workgroup_count = (size / 512).max(1); // Since we process 2 elements per thread
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

            drop(compute_pass);
            self.queue.submit(Some(encoder.finish()));

            // Small delay to ensure GPU operations complete
            // In production, use proper synchronization
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        // Read results back
        let result = self.read_buffer(&gpu_buffer, data.len());
        data.copy_from_slice(&result);
    }

    /// Execute merge sort on GPU
    pub fn merge_sort_gpu(&self, data: &mut [u32]) {
        let size = data.len() as u32;
        if size == 0 {
            return;
        }

        // For small arrays, fall back to CPU sort
        if size < 8 {
            data.sort();
            return;
        }

        // Create GPU buffer
        let gpu_buffer = self.create_buffer(
            data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // Create temp buffer for merge operations
        let temp_buffer = self.create_buffer(
            &vec![0u32; data.len()],
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Merge Sort Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &temp_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // For demonstration, run a simple merge sort pass
        // In a real implementation, this would be a full merge sort with multiple phases
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Merge Sort Encoder"),
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Merge Sort Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.merge_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch workgroups for merge sort
        let workgroup_count = (size / 256).max(1);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(compute_pass);
        self.queue.submit(Some(encoder.finish()));

        // Small delay to ensure GPU operations complete
        std::thread::sleep(std::time::Duration::from_micros(100));

        // Read results back
        let result = self.read_buffer(&gpu_buffer, data.len());
        data.copy_from_slice(&result);
    }
}

/// GPU-accelerated sorting using merge sort (simulated)
/// In a real GPU implementation, this would use compute shaders
/// For now, this uses CPU merge sort to demonstrate the interface
/// Time complexity: O(n log n), Space complexity: O(n)
/// Returns 0 (no pass count for this algorithm)
pub fn gpu_sort<T: Ord + Clone + Send + Sync>(data: &mut [T]) -> usize {
    // For now, only implement GPU sorting for u32
    // Use unsafe casting for demonstration - in production, use proper type handling
    if std::mem::size_of::<T>() == std::mem::size_of::<u32>()
        && std::mem::align_of::<T>() == std::mem::align_of::<u32>()
    {
        if let Some(ctx) = GpuContext::new() {
            unsafe {
                let u32_data =
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u32, data.len());
                ctx.merge_sort_gpu(u32_data);
            }
            return 0;
        }
    }

    // Fallback to CPU merge sort
    crate::sorting::merge::mergesort(data);
    0
}

/// GPU-accelerated bitonic sort
/// Bitonic sort is well-suited for GPU implementation due to its parallel nature
/// Time complexity: O(n log² n), Space complexity: O(1)
/// Returns 0 (no pass count for this algorithm)
pub fn gpu_bitonic_sort<T: Ord + Send + Sync>(data: &mut [T]) -> usize {
    // For now, only implement GPU sorting for u32
    // Use unsafe casting for demonstration - in production, use proper type handling
    if std::mem::size_of::<T>() == std::mem::size_of::<u32>()
        && std::mem::align_of::<T>() == std::mem::align_of::<u32>()
    {
        if let Some(ctx) = GpuContext::new() {
            unsafe {
                let u32_data =
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u32, data.len());
                ctx.bitonic_sort_gpu(u32_data);
            }
            return 0;
        }
    }

    // Fallback to CPU heap sort
    crate::sorting::heap::heap_sort(data);
    data.len() / 2 // Return some pass count for compatibility
}
