//! Controlled lease-storage experiments. Run explicitly in release mode.
//! Set ICED_LEASE_BENCH_VARIANTS=inline to record the pre-change B1 baseline.

use super::*;
use std::hint::black_box;
use std::time::Instant;

mod resident;
use resident::Resident;

trait Resource {
    fn key(&self) -> Key;
    fn role(&self) -> Role;
    fn set_role(&mut self, role: Role);
    fn touch(&mut self) -> u64;
}

struct CpuTarget {
    key: Key,
    role: Role,
    id: u64,
    payload: [u64; 27],
}

impl Resource for CpuTarget {
    fn key(&self) -> Key {
        self.key
    }
    fn role(&self) -> Role {
        self.role
    }
    fn set_role(&mut self, role: Role) {
        self.role = role;
    }
    fn touch(&mut self) -> u64 {
        self.payload[0] = self.payload[0].wrapping_add(1);
        black_box(self.id ^ self.payload[0] ^ self.payload[13] ^ self.payload[26])
    }
}

impl Resource for TargetStorage {
    fn key(&self) -> Key {
        Key::new(self.format, self.extent)
    }
    fn role(&self) -> Role {
        self.last_role
    }
    fn set_role(&mut self, role: Role) {
        self.last_role = role;
    }
    fn touch(&mut self) -> u64 {
        if self.last_role == Role::Scratch {
            // Like ScratchAllocator: a caller may retain a cloned view, but the
            // clone does not release its lease early. No GPU commands are run.
            let view = black_box(self.view.clone());
            black_box(u64::from(view.texture().size().width))
        } else {
            black_box(self.byte_size())
        }
    }
}

trait Storage<T: Resource>: Default {
    type Lease;
    fn take(&mut self, key: Key) -> Option<Self::Lease>;
    fn insert(&mut self, target: T) -> Self::Lease;
    fn get<'a>(&'a self, lease: &'a Self::Lease) -> &'a T;
    fn get_mut<'a>(&'a mut self, lease: &'a mut Self::Lease) -> &'a mut T;
    fn release(&mut self, lease: Self::Lease);
    fn trim(&mut self, allowance: u64, budget: u64) -> usize;
    fn free(&self) -> (u64, u64);
    // Owned capacities only: excludes hash control bytes and allocator headers.
    fn heap_bytes(&self, held: usize) -> usize;
}

fn free_heap<T>(pool: &FreePool<T>) -> usize {
    pool.buckets.capacity() * std::mem::size_of::<(Key, Bucket<T>)>()
        + pool
            .buckets
            .values()
            .map(|bucket| bucket.entries.capacity() * std::mem::size_of::<Free<T>>())
            .sum::<usize>()
        + pool.candidates.capacity() * std::mem::size_of::<(Key, u64)>()
}

impl<T: Resource> Storage<T> for FreePool<T> {
    type Lease = T;
    fn take(&mut self, key: Key) -> Option<T> {
        self.take(key)
    }
    fn insert(&mut self, target: T) -> T {
        target
    }
    fn get<'a>(&'a self, lease: &'a T) -> &'a T {
        lease
    }
    fn get_mut<'a>(&'a mut self, lease: &'a mut T) -> &'a mut T {
        lease
    }
    fn release(&mut self, target: T) {
        self.release(target.key(), target.role(), target);
    }
    fn trim(&mut self, allowance: u64, budget: u64) -> usize {
        self.trim_to_bytes(allowance, budget).total
    }
    fn free(&self) -> (u64, u64) {
        (self.bytes, self.scratch_bytes)
    }
    fn heap_bytes(&self, _: usize) -> usize {
        free_heap(self)
    }
}

#[derive(Clone, Copy)]
struct Workload {
    name: &'static str,
    count: usize,
    classes: usize,
    resizing: bool,
    pressure: bool,
    held: bool,
    burst: bool,
}

struct Frame {
    keys: Vec<Key>,
    returns: Vec<usize>,
}

struct Boxed<T> {
    free: FreePool<Lease<T>>,
}

impl<T> Default for Boxed<T> {
    fn default() -> Self {
        Self {
            free: FreePool::default(),
        }
    }
}

impl<T: Resource> Storage<T> for Boxed<T> {
    type Lease = Lease<T>;
    fn take(&mut self, key: Key) -> Option<Lease<T>> {
        self.free.take(key)
    }
    fn insert(&mut self, target: T) -> Lease<T> {
        Lease(Box::new(target))
    }
    fn get<'a>(&'a self, lease: &'a Lease<T>) -> &'a T {
        lease
    }
    fn get_mut<'a>(&'a mut self, lease: &'a mut Lease<T>) -> &'a mut T {
        lease
    }
    fn release(&mut self, target: Lease<T>) {
        self.free.release(target.key(), target.role(), target);
    }
    fn trim(&mut self, allowance: u64, budget: u64) -> usize {
        self.free.trim_to_bytes(allowance, budget).total
    }
    fn free(&self) -> (u64, u64) {
        (self.free.bytes, self.free.scratch_bytes)
    }
    fn heap_bytes(&self, held: usize) -> usize {
        let count = self
            .free
            .buckets
            .values()
            .map(|bucket| bucket.entries.len())
            .sum::<usize>();
        free_heap(&self.free) + (count + held) * std::mem::size_of::<T>()
    }
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn shuffle<T>(values: &mut [T], state: &mut u64) {
    for index in (1..values.len()).rev() {
        values.swap(index, random(state) as usize % (index + 1));
    }
}

fn trace(workload: Workload, seed: u64) -> Vec<Frame> {
    let mut state = seed;
    (0..288)
        .map(|frame| {
            let count = if workload.burst && frame % 64 >= 8 {
                0
            } else {
                workload.count
            };
            let mut keys: Vec<_> = (0..count)
                .map(|index| {
                    let class = (index * 17 + if workload.resizing { frame * 7 } else { 0 })
                        % workload.classes;
                    Key::new(
                        wgpu::TextureFormat::Rgba8Unorm,
                        Size::new(64 * (1 + class as u32 % 8), 64 * (1 + class as u32 / 8)),
                    )
                })
                .collect();
            shuffle(&mut keys, &mut state);
            let mut returns: Vec<_> = (0..count).collect();
            shuffle(&mut returns, &mut state);
            Frame { keys, returns }
        })
        .collect()
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Counts {
    allocations: usize,
    hits: usize,
    trims: usize,
    checksum: u64,
    free_bytes: u64,
    scratch_bytes: u64,
}

struct Sample {
    frames_ns: Vec<f64>,
    counts: Counts,
    peak_heap: usize,
}

fn measure<T: Resource, S: Storage<T>>(
    workload: Workload,
    trace: &[Frame],
    mut create: impl FnMut(Key, u64) -> T,
) -> Sample {
    let mut pool = S::default();
    let mut active: Vec<Option<S::Lease>> = Vec::with_capacity(workload.count);
    let mut held: Vec<Option<(usize, S::Lease)>> = Vec::new();
    let mut next_id = 0;
    let mut counts = Counts::default();
    let mut times = Vec::with_capacity(256);
    let mut peak_heap = 0;
    let budget = if workload.pressure {
        trace[0].keys.iter().map(|key| key.bytes()).sum::<u64>() / 4
    } else {
        u64::MAX / 2
    };
    for (frame_number, frame) in trace.iter().enumerate() {
        if frame_number == 32 {
            counts = Counts::default();
        }
        let started = Instant::now();
        for entry in &mut held {
            if entry.as_ref().is_some_and(|(due, _)| *due <= frame_number) {
                let (_, lease) = entry.take().unwrap();
                pool.release(lease);
            }
        }
        for (index, &key) in frame.keys.iter().enumerate() {
            let mut lease = if let Some(lease) = pool.take(black_box(key)) {
                counts.hits += 1;
                lease
            } else {
                counts.allocations += 1;
                next_id += 1;
                pool.insert(create(key, next_id))
            };
            let target = pool.get_mut(&mut lease);
            target.set_role(if (index + frame_number).is_multiple_of(3) {
                Role::Scratch
            } else {
                Role::Layer
            });
            counts.checksum = counts.checksum.wrapping_add(target.touch());
            active.push(Some(lease));
        }
        // A separate access pass models prepared/rendered target access while
        // all leases are outstanding. No bulk rotation of large target values.
        for lease in active.iter_mut().flatten() {
            counts.checksum = counts.checksum.wrapping_add(pool.get_mut(lease).touch());
        }
        for &index in &frame.returns {
            let lease = active[index].take().unwrap();
            if workload.held && index.is_multiple_of(8) {
                let entry = Some((frame_number + 4, lease));
                if let Some(slot) = held.iter_mut().find(|entry| entry.is_none()) {
                    *slot = entry;
                } else {
                    held.push(entry);
                }
            } else {
                pool.release(lease);
            }
        }
        active.clear();
        let held_bytes: u64 = held
            .iter()
            .flatten()
            .map(|(_, lease)| pool.get(lease).key().bytes())
            .sum();
        counts.trims += pool.trim(budget.saturating_sub(held_bytes), budget);
        (counts.free_bytes, counts.scratch_bytes) = black_box(pool.free());
        if frame_number >= 32 {
            times.push(started.elapsed().as_secs_f64() * 1e9);
        }
        // Capacity scans are diagnostic, outside the timed frame.
        let heap = pool.heap_bytes(held.iter().flatten().count())
            + active.capacity() * std::mem::size_of::<Option<S::Lease>>()
            + held.capacity() * std::mem::size_of::<Option<(usize, S::Lease)>>();
        peak_heap = peak_heap.max(heap);
    }
    for (_, lease) in held.into_iter().flatten() {
        pool.release(lease);
    }
    Sample {
        frames_ns: times,
        counts,
        peak_heap,
    }
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Inline,
    Boxed,
    Resident,
}

fn run<T: Resource>(
    variant: Variant,
    workload: Workload,
    trace: &[Frame],
    create: impl FnMut(Key, u64) -> T,
) -> Sample {
    match variant {
        Variant::Inline => measure::<T, FreePool<T>>(workload, trace, create),
        Variant::Boxed => measure::<T, Boxed<T>>(workload, trace, create),
        Variant::Resident => measure::<T, Resident<T>>(workload, trace, create),
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[(sorted.len() as f64 * fraction).ceil() as usize - 1]
}

fn compare<T: Resource>(
    workloads: &[Workload],
    samples: usize,
    mut create: impl FnMut(Key, u64) -> T,
) {
    let variants = match std::env::var("ICED_LEASE_BENCH_VARIANTS").as_deref() {
        Ok("inline") => vec![Variant::Inline],
        Err(std::env::VarError::NotPresent) | Ok("all") => {
            vec![Variant::Inline, Variant::Boxed, Variant::Resident]
        }
        _ => panic!("ICED_LEASE_BENCH_VARIANTS must be inline or all"),
    };
    for &workload in workloads {
        let mut means = vec![Vec::new(); variants.len()];
        let mut times = vec![Vec::new(); variants.len()];
        let mut heaps = vec![0; variants.len()];
        let mut totals = vec![Counts::default(); variants.len()];
        for sample in 0..samples {
            let trace = trace(workload, 0xcafe_f00d + sample as u64 * 1337);
            let mut expected = None;
            for position in 0..variants.len() {
                let index = (position + sample) % variants.len();
                let result = run(variants[index], workload, &trace, &mut create);
                if let Some(expected) = &expected {
                    assert_eq!(
                        &result.counts, expected,
                        "same trace must produce identical ownership/results"
                    );
                }
                expected = Some(result.counts.clone());
                means[index]
                    .push(result.frames_ns.iter().sum::<f64>() / result.frames_ns.len() as f64);
                times[index].extend(result.frames_ns);
                heaps[index] = heaps[index].max(result.peak_heap);
                totals[index].allocations += result.counts.allocations;
                totals[index].hits += result.counts.hits;
                totals[index].trims += result.counts.trims;
                totals[index].checksum =
                    totals[index].checksum.wrapping_add(result.counts.checksum);
                totals[index].free_bytes = result.counts.free_bytes;
                totals[index].scratch_bytes = result.counts.scratch_bytes;
            }
        }
        for (index, variant) in variants.iter().enumerate() {
            means[index].sort_by(f64::total_cmp);
            times[index].sort_by(f64::total_cmp);
            eprintln!(
                "{} {:?}: mean_ns={:.0} frame_p95={:.0} frame_p99={:.0} post_frame_heap_estimate={} {:?}",
                workload.name,
                variant,
                percentile(&means[index], 0.5),
                percentile(&times[index], 0.95),
                percentile(&times[index], 0.99),
                heaps[index],
                totals[index]
            );
        }
    }
}

fn workloads() -> [Workload; 8] {
    let base = Workload {
        name: "tiny-hot",
        count: 16,
        classes: 4,
        resizing: false,
        pressure: false,
        held: false,
        burst: false,
    };
    [
        base,
        Workload {
            name: "repeated",
            count: 512,
            classes: 8,
            ..base
        },
        Workload {
            name: "resizing",
            count: 128,
            classes: 512,
            resizing: true,
            ..base
        },
        Workload {
            name: "churn-pressure",
            count: 128,
            classes: 4096,
            resizing: true,
            pressure: true,
            ..base
        },
        Workload {
            name: "few-key-pressure",
            count: 512,
            classes: 8,
            pressure: true,
            ..base
        },
        Workload {
            name: "wide-pressure",
            count: 512,
            classes: 512,
            resizing: true,
            pressure: true,
            ..base
        },
        Workload {
            name: "delayed-returns",
            count: 128,
            classes: 16,
            held: true,
            ..base
        },
        Workload {
            name: "burst-idle",
            count: 512,
            classes: 128,
            burst: true,
            ..base
        },
    ]
}

#[test]
#[ignore = "manual release-mode lease-storage CPU benchmark"]
fn cpu_lease_storage_benchmark() {
    if cfg!(debug_assertions) {
        panic!("run with --release");
    }
    eprintln!(
        "CPU resource={} backing={} bytes; 21 seeds x 256 measured frames; 32 warmup; true frame p95/p99",
        std::mem::size_of::<CpuTarget>(),
        std::mem::size_of::<TargetStorage>()
    );
    assert_eq!(
        std::mem::size_of::<CpuTarget>(),
        std::mem::size_of::<TargetStorage>()
    );
    compare(&workloads(), 21, |key, id| CpuTarget {
        key,
        role: Role::Layer,
        id,
        payload: [id; 27],
    });
}

#[test]
#[ignore = "manual release-mode benchmark requiring a native GPU"]
fn gpu_lease_storage_benchmark() {
    if cfg!(debug_assertions) {
        panic!("run with --release");
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = futures::executor::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )
    .unwrap();
    let (device, _queue) =
        futures::executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .unwrap();
    eprintln!(
        "GPU resources, CPU-only timed reuse: {:?}",
        adapter.get_info()
    );
    compare(&workloads()[..2], 11, |key, _| {
        TargetStorage::new(
            &device,
            key.format,
            Size::new(key.width, key.height),
            Role::Layer,
        )
    });
}
