//! Run with `cargo test -p iced_wgpu --release --lib cpu_pool_before_after
//! -- --ignored --nocapture`. CPU policy costs only, not GPU allocation costs.
//! Historical vector versus inline-B1 control. Production target leases are
//! now boxed; use `lease_storage_benchmark` to compare storage representations.

use super::super::*;
use std::hint::black_box;
use std::time::Instant;

// Both implementations move identical stand-ins sized like the native Target.
// No timing here measures wgpu handle destruction or driver work.
struct CpuTarget {
    metadata: Metadata,
    payload: [usize; PAYLOAD_WORDS],
}

struct Metadata {
    key: Key,
    role: Role,
    last_used: u64,
}

const PAYLOAD_WORDS: usize = std::mem::size_of::<TargetStorage>()
    .saturating_sub(std::mem::size_of::<Metadata>())
    .div_ceil(std::mem::size_of::<usize>());

trait BenchPool: Default {
    fn take(&mut self, key: Key, frame: u64) -> Option<CpuTarget>;
    fn release(&mut self, target: CpuTarget, frame: u64);
    fn finish(&mut self, frame: u64, budget: u64) -> usize;
    fn usage(&self) -> (u64, u64);
}

impl BenchPool for FreePool<CpuTarget> {
    fn take(&mut self, key: Key, _: u64) -> Option<CpuTarget> {
        self.take(key)
    }
    fn release(&mut self, target: CpuTarget, _: u64) {
        self.release(target.metadata.key, target.metadata.role, target);
    }
    fn finish(&mut self, _: u64, budget: u64) -> usize {
        self.trim_to_bytes(budget, budget).total
    }
    fn usage(&self) -> (u64, u64) {
        (self.bytes, self.scratch_bytes)
    }
}

/// Algorithm from the pre-change pool at f0fa7cd: first-match swap_remove,
/// frame/role eviction ties, repeated byte scans, and the 120-frame idle TTL.
#[derive(Default)]
struct Before {
    free: Vec<CpuTarget>,
}

impl Before {
    fn bytes(&self) -> u64 {
        self.free.iter().fold(0, |bytes, target| {
            bytes.saturating_add(target.metadata.key.bytes())
        })
    }
}

impl BenchPool for Before {
    fn take(&mut self, key: Key, frame: u64) -> Option<CpuTarget> {
        let index = self
            .free
            .iter()
            .position(|target| target.metadata.key == key)?;
        let mut target = self.free.swap_remove(index);
        target.metadata.last_used = frame;
        Some(target)
    }
    fn release(&mut self, mut target: CpuTarget, frame: u64) {
        target.metadata.last_used = frame;
        self.free.push(target);
    }
    fn finish(&mut self, frame: u64, budget: u64) -> usize {
        let initial = self.free.len();
        self.free
            .retain(|target| frame.wrapping_sub(target.metadata.last_used) <= 120);
        while self.bytes() > budget {
            let index = self
                .free
                .iter()
                .enumerate()
                .max_by_key(|(_, target)| {
                    (
                        frame.wrapping_sub(target.metadata.last_used),
                        target.metadata.role == Role::Layer,
                    )
                })
                .map(|(index, _)| index)
                .unwrap();
            let _ = self.free.swap_remove(index);
        }
        initial - self.free.len()
    }
    fn usage(&self) -> (u64, u64) {
        (
            self.bytes(),
            self.free
                .iter()
                .filter(|target| target.metadata.role == Role::Scratch)
                .fold(0, |bytes, target| {
                    bytes.saturating_add(target.metadata.key.bytes())
                }),
        )
    }
}

struct Workload {
    name: &'static str,
    frames: Vec<Vec<Key>>,
    budget: u64,
}

fn workload(
    name: &'static str,
    count: usize,
    classes: usize,
    resizing: bool,
    pressure: bool,
) -> Workload {
    let frames: Vec<Vec<_>> = (0..256)
        .map(|frame| {
            (0..count)
                .map(|index| {
                    let class = (index * 17 + if resizing { frame * 7 } else { 0 }) % classes;
                    Key::new(
                        wgpu::TextureFormat::Rgba8Unorm,
                        Size::new(64 * (1 + class as u32 % 32), 64 * (1 + class as u32 / 32)),
                    )
                })
                .collect()
        })
        .collect();
    let budget = if pressure {
        frames[0].iter().map(|key| key.bytes()).sum::<u64>() / 4
    } else {
        u64::MAX / 2
    };
    Workload {
        name,
        frames,
        budget,
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    allocations: usize,
    trims: usize,
    final_bytes: u64,
}

fn measure<P: BenchPool>(workload: &Workload) -> (f64, Counts) {
    let mut pool = P::default();
    let mut active = Vec::with_capacity(workload.frames[0].len());
    let mut counts = Counts::default();
    // Warm metadata and backings outside the measured interval.
    for (index, keys) in workload.frames.iter().take(32).enumerate() {
        frame(
            &mut pool,
            &mut active,
            keys,
            index as u64,
            workload.budget,
            &mut counts,
        );
    }
    counts = Counts::default();
    let started = Instant::now();
    for (index, keys) in workload.frames.iter().enumerate() {
        frame(
            &mut pool,
            &mut active,
            keys,
            index as u64 + 32,
            workload.budget,
            &mut counts,
        );
    }
    let elapsed = started.elapsed().as_secs_f64() * 1e9 / workload.frames.len() as f64;
    counts.final_bytes = black_box(pool.usage().0);
    (elapsed, counts)
}

fn frame<P: BenchPool>(
    pool: &mut P,
    active: &mut Vec<CpuTarget>,
    keys: &[Key],
    frame: u64,
    budget: u64,
    counts: &mut Counts,
) {
    for (index, &key) in keys.iter().enumerate() {
        let mut target = pool.take(black_box(key), frame).unwrap_or_else(|| {
            counts.allocations += 1;
            CpuTarget {
                metadata: Metadata {
                    key,
                    role: Role::Layer,
                    last_used: frame,
                },
                payload: [index; PAYLOAD_WORDS],
            }
        });
        target.metadata.role = if (index + frame as usize).is_multiple_of(3) {
            Role::Scratch
        } else {
            Role::Layer
        };
        let _ = black_box(&mut target.payload);
        active.push(target);
    }
    // Simultaneous leases followed by a rotating return order, not an artificial
    // lease/return pair that lets every request reuse one backing.
    let offset = frame as usize % active.len();
    active.rotate_left(offset);
    while let Some(target) = active.pop() {
        pool.release(target, frame);
    }
    counts.trims += pool.finish(frame, budget);
    let _ = black_box(pool.usage());
}

#[test]
#[ignore = "manual release-mode CPU benchmark, not a correctness test"]
fn cpu_pool_before_after() {
    if cfg!(debug_assertions) {
        panic!("run this benchmark with --release");
    }
    eprintln!(
        "stand-in={} B; GPU Target={} B; 21 alternating paired samples; batch-mean ns/frame median/p95",
        std::mem::size_of::<CpuTarget>(),
        std::mem::size_of::<TargetStorage>()
    );
    for workload in [
        workload("tiny-hot", 16, 4, false, false),
        workload("repeated-sizes", 512, 8, false, false),
        workload("resizing", 128, 512, true, false),
        workload("churn-pressure", 128, 512, true, true),
        workload("few-class-pressure", 512, 8, false, true),
        workload("wide-pressure", 512, 512, true, true),
    ] {
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut counts = None;
        for sample in 0..21 {
            let (old, new) = if sample % 2 == 0 {
                (
                    measure::<Before>(&workload),
                    measure::<FreePool<CpuTarget>>(&workload),
                )
            } else {
                let new = measure::<FreePool<CpuTarget>>(&workload);
                (measure::<Before>(&workload), new)
            };
            before.push(old.0);
            after.push(new.0);
            let pair = (old.1, new.1);
            if let Some(expected) = &counts {
                assert_eq!(&pair, expected);
            }
            counts = Some(pair);
        }
        before.sort_by(f64::total_cmp);
        after.sort_by(f64::total_cmp);
        eprintln!(
            "{}: before {:.0}/{:.0}, after {:.0}/{:.0}, ratio {:.2}; counts {:?}",
            workload.name,
            before[10],
            before[19],
            after[10],
            after[19],
            after[10] / before[10],
            counts.unwrap()
        );
    }
}
