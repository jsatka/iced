# Isolated layer texture pooling

Current implementation baseline, 2026-09-07: **B1 compatibility buckets with
boxed owning leases**. This document describes the implemented behavior, not a
pending design plan. The [completed storage comparison](isolated-layer-lease-storage-experiment.md)
records the architecture decision, reproducible measurements, and validation.

## Ownership and lookup

[`Pool`](../wgpu/src/isolated_layer/pool.rs) owns reusable standalone texture
backings through `FreePool<Target>`. Its `FxHashMap<Key, Bucket<T>>` groups free
backings by exact pixel width, pixel height, and texture format. Each bucket has
an oldest-to-newest `VecDeque`; checkout takes the newest compatible backing,
and return appends it. Layer and scratch allocations share compatible backings.
The last lease role affects diagnostics, not compatibility.

`Target` is a private alias for `Lease<TargetStorage>`, a non-cloneable owning
wrapper around `Box<TargetStorage>`. Only the pointer-sized lease moves between
the pool, prepared nodes, scratch allocator, and retained registry. The storage
stays at a stable CPU address. On the measured native platform, the lease is
8 bytes and its storage is 248 bytes. Moving either representation moves CPU
bookkeeping, not GPU pixels.

There is one additional CPU heap allocation per newly created backing, not per
reuse. Ordinary `Deref`/`DerefMut` accesses the storage. Exclusive ownership
prevents simultaneous free and active membership without a leased-target map,
reference counting, locks, or unsafe code. Dropping a lease destroys its owned
storage normally; the existing scratch unwind guard returns scratch leases.
Public effect and scratch APIs are unchanged.

This is deliberately not `lease(&mut self) -> &Target`. Long-lived pool borrows
would conflict with subsequent leases and recursive preparation while a parent's
text viewport is borrowed. The checked resident-slot implementation exists only
in the benchmark; it is not part of renderer ownership.

### Why the key stores pixel dimensions

The layer-specific `SizeClass` is not a complete compatibility key for the shared
pool. Layer dimensions use 64-pixel increments, but scratch dimensions up to 64
use powers of two; device-limit clamping can also produce non-64-aligned extents.
An 8-by-8 scratch backing and a 64-by-64 backing must not collide. Equal area is
also insufficient. Exact width, height, and format preserve these distinctions.

One mip, one sample, 2D single-layer storage, and fixed usage flags are common
invariants. If those become variable, the compatibility key must expand.
Reducing a key's byte size would not alone prove cheaper hashing: derived hashes
process fields rather than raw struct storage or padding.

## Budget and eviction contract

The configured memory budget is a retention threshold, not a preallocation
target or a hard cap on all GPU memory. At or below it, unused free textures
remain available indefinitely; there is no idle TTL. Defaults remain 128 MiB
natively and 32 MiB on WebAssembly.

[`State`](../wgpu/src/isolated_layer/mod.rs) enforces the shared free-plus-retained
ownership budget at frame end, after normal cleanup. Allocation can exceed the
threshold during drawing. Eligible ownership is trimmed in the same frame, with
no deferred cleanup, low-water mark, or hysteresis:

1. Audit retained leases and perform the normal retained liveness sweep, returning
   released backings to the free pool.
2. If accounted ownership fits, skip pressure eviction and retained candidate
   sorting. Empty-bucket metadata cleanup still occurs.
3. Under pressure, discard free backings individually larger than the **entire
   configured budget** first. The cutoff is not the smaller free allowance left
   after retained ownership.
4. Evict remaining free backings in global oldest-return order until the free
   allowance is satisfied or no free backing remains.
5. If still over budget, evict eligible normal cached outputs in oldest-access
   order, then protected outputs in oldest-access order.

Active leases are ineligible. If they prevent compliance, existing total-byte
and unfinished-lease diagnostics expose the excess; debug assertions require
all eligible ownership to have been reclaimed. Existing abandoned-lease recovery
makes those retained slots eligible again on the next frame.

Whole-backing eviction can undershoot the threshold. Apart from the individually
oversized exception, eviction is not largest-first or best-fit. Pressure victims
leave renderer ownership rather than returning to the same free pool.
Trimming drops ownership; it never calls `Texture::destroy`, because encoded
commands may still reference a texture before submission.

### Recency, cached content, and protection

Free recency means when a backing became available for reuse, not when the GPU
last accessed it. Each return gets a pool-wide sequence number, distinguishing
same-frame returns across buckets. Before overflow, live stamps are rebased
without changing their order. MRU checkout concentrates reuse on warm backings
and leaves surplus backings cold for eviction.

The [retained registry](../wgpu/src/isolated_layer/retained.rs) keeps its separate
rendered-frame access recency and liveness rules. Keep-alive refreshes liveness,
not access recency. A backing returned by liveness cleanup receives fresh free
recency even if its former cached pixels had not been sampled recently.

[`CacheResidencyPriority::Protected`](../core/src/isolated_layer/cache.rs) is the
last eligible pressure tier, not an unbreakable pin. Protected content can still
expire through liveness or be evicted after lower-priority eligible ownership.
Holding a surface handle alone does not guarantee residency. The retained grace
period remains two rendered frames by default. Liveness can forget content
below budget while its reusable backing remains pooled indefinitely.

### Accounting boundaries

Free total and scratch bytes are maintained incrementally. Checkout subtracts
the previous role's contribution before changing the role; return adds the
current contribution. Budget trim diagnostics use the last lease role. Public
`pool_idle_trims` and `scratch_idle_trims` remain for compatibility and are always
zero under this policy.

Byte estimates use actual backing extent and the format's texel-block size.
They are not measured physical VRAM. Driver alignment/metadata, CPU metadata,
optional text viewport resources, independent private allocations, active draw
allocations outside the accounted retained slots, and command-only references
after ownership release are not a total-memory cap. Dropping ownership does not
promise immediate driver reclamation.

## Bookkeeping costs and capacity

Ordinary exact checkout and return are expected amortized `O(1)`; free byte
queries are `O(1)`. Under pressure, a reusable dense candidate vector contains
one oldest entry per nonempty bucket. Each victim selection scans those heads,
pops one backing, and updates or removes that candidate. It does not drain an
entire selected class ahead of older backings in other classes.

With hash-table capacity `H`, nonempty key count `K`, and `E` free victims,
ordinary global-oldest selection costs `O(H + E K)`. Broad trims can therefore
still be quadratic when most backings have distinct keys.

Empty bucket deques survive checkout/return cycles within a frame. Empty buckets
are removed at frame end, an `O(H)` scan even below budget. Hash-table, candidate
vector, and nonempty deque capacities are retained without automatic shrinking.
This avoids repeated metadata allocation but retains CPU high-water capacity;
that capacity is not part of the texture byte budget.

## Measured performance and validation

The [storage report](isolated-layer-lease-storage-experiment.md) is the canonical
measurement record. Three independent paired runs compare the current boxed
implementation against inline B1 and a test-only resident-slot prototype using
identical event traces and matching result assertions.

Boxed leases reduced mean warmed CPU reuse time by about 45–50% in the synthetic
traces and about 54–55% with real native GPU target objects. Pressure-heavy mean
improvements were smaller, about 2.7–15.3%; individual-frame tails were not
uniformly better. These are pool-bookkeeping results, not renderer/GPU speedups.
Real-object measurements cover warm reuse, not GPU allocation churn.

The older `cpu_pool_before_after` test is a historical vector-versus-inline-B1
control, not a measurement of current production storage. Its few-key regression
does not describe boxed B1, and numbers from the two different harnesses must
not be combined into a claimed boxed-versus-vector speedup.

Validation of this implementation on Apple M2 Pro / Metal included 73 ordinary
library tests and nine explicit native GPU tests. Coverage includes a 10,000-event
reference-model check, MRU/LRU and role accounting, oversized and zero budgets,
indefinite retention, sequence rebasing, cache priority, active-slot recovery,
stable lease addresses, exactly-once destruction, scratch unwind, poisoned
backing rendering, and encoded-command lifetime after ownership release.

Native all-feature and wasm32 WebGPU/WebGL compilation checks passed. Clippy
completed with two pre-existing warnings. Browser runtime and performance were
not tested. Environment details and reproduction commands are in the report.

## Alternatives not included in this baseline

- **Linear collections:** short successful vector scans can be cheap for few hot
  keys, but miss search and ordering maintenance remain linear. The current
  baseline keeps indexed exact lookup and boxed storage.
- **Temporary head heap:** reduces broad-trim selection to
  `O(H + E log(K + 1))`, with additional setup and temporary bookkeeping. It is
  not implemented; B1 uses dense head scans.
- **Persistent global/per-key ordering:** can avoid head scans but adds links and
  updates on every lease/return. The test-only resident-slot experiment addresses
  storage, not an implemented persistent global LRU index.
- **Texture atlases:** independent deallocation is feasible with dynamic rectangle
  allocators; churn alone does not rule them out. However, evicting a rectangle
  need not free a page, and sparse protected survivors can strand page capacity.
  Current whole-view clears, raw custom-effect views, copy coordinates, and
  sampling/render-attachment subresource constraints prevent a transparent
  rectangular-atlas replacement. Standalone backings preserve per-texture
  reclamation and existing APIs. The existing [image atlas](../wgpu/src/image/atlas.rs)
  is not a drop-in render-target allocator.

MRU backing reuse paired with oldest eligible eviction has precedents in
[Chromium's resource pool](https://raw.githubusercontent.com/chromium/chromium/main/cc/resources/resource_pool.cc)
and [Skia's resource cache](https://skia.googlesource.com/skia/+/refs/heads/main/src/gpu/ganesh/GrResourceCache.cpp).
These informed the architecture, not performance claims for this fork.
[Guillotière](https://github.com/nical/guillotiere) and
[Étagère](https://docs.rs/etagere/latest/etagere/) provide dynamic atlas allocation
precedents; neither establishes a win for this renderer's lifetime mix.
These sources were inspected during the design evaluation on 2026-09-07.

## Remaining evidence gaps

The implementation and policy choices are settled for this baseline. No pending
design interview or alternative implementation is required before committing it.
The following limits remain relevant to interpreting the evidence:

- No measured application workload distribution or end-to-end renderer profile.
- No browser runtime/performance results or representative cross-device timings.
- No real-GPU allocation-churn or driver-reclamation timing in the storage tests.
- No full renderer allocator profile, including persistent CPU capacity after
  bursts, or established broad-trim latency requirement.
- No atlas trace/fragmentation evidence or approved region-aware effect contract.

Any later change to exact matching, same-frame trimming, indefinite under-budget
free retention, or protected-last eligibility is a separate policy decision.
