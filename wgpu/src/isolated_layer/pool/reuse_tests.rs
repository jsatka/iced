//! CPU-only checks of the same free-pool implementation used with GPU targets.

use super::*;

#[cfg(not(target_arch = "wasm32"))]
mod benchmark;

fn key(width: u32) -> Key {
    Key::new(wgpu::TextureFormat::Rgba8Unorm, Size::new(width, 1))
}

fn ids(pool: &FreePool<usize>) -> Vec<usize> {
    let mut ids: Vec<_> = pool
        .buckets
        .values()
        .flat_map(|bucket| bucket.entries.iter().map(|entry| entry.target))
        .collect();
    ids.sort_unstable();
    ids
}

fn assert_usage(pool: &FreePool<usize>) {
    let mut bytes = 0;
    let mut scratch = 0;
    for bucket in pool.buckets.values() {
        let mut previous = None;
        for entry in &bucket.entries {
            bytes += bucket.bytes;
            if entry.role == Role::Scratch {
                scratch += bucket.bytes;
            }
            assert!(previous.is_none_or(|stamp| stamp < entry.released_at));
            previous = Some(entry.released_at);
        }
    }
    assert_eq!((pool.bytes, pool.scratch_bytes), (bytes, scratch));
    assert!(pool.candidates.is_empty());
}

#[test]
fn newest_exact_match_is_reused_and_active_entries_are_excluded() {
    let mut pool = FreePool::default();
    pool.release(key(4), Role::Layer, 1);
    pool.release(key(4), Role::Scratch, 2);
    pool.release(key(8), Role::Scratch, 3);
    let float = Key {
        format: wgpu::TextureFormat::Rgba16Float,
        ..key(4)
    };
    assert_eq!(pool.take(float), None);
    assert_eq!(pool.take(key(4)), Some(2));
    assert_eq!((pool.bytes, pool.scratch_bytes), (48, 32));
    assert_eq!(pool.take(key(4)), Some(1));
    assert_eq!(pool.take(key(4)), None);
    assert_eq!(pool.take(key(8)), Some(3));
    assert_eq!(pool.bytes, 0);
    // A role change on a new lease changes accounting only after return.
    pool.release(key(4), Role::Layer, 2);
    assert_eq!(pool.scratch_bytes, 0);
    assert_usage(&pool);
}

#[test]
fn lru_is_per_backing_across_buckets_and_return_refreshes_recency() {
    let mut pool = FreePool::default();
    pool.release(key(4), Role::Scratch, 1);
    pool.release(key(8), Role::Layer, 2);
    pool.release(key(4), Role::Layer, 3);
    // Do not discard all of class 4 just because its front is the oldest.
    assert_eq!(
        pool.trim_to_bytes(48, 64),
        Trimmed {
            total: 1,
            scratch: 1
        }
    );
    assert_eq!(ids(&pool), [2, 3]);
    let target = pool.take(key(8)).unwrap();
    pool.release(key(8), Role::Scratch, target);
    assert_eq!(
        pool.trim_to_bytes(32, 64),
        Trimmed {
            total: 1,
            scratch: 0
        }
    );
    assert_eq!(ids(&pool), [2]);
    assert_usage(&pool);
}

#[test]
fn oversized_backings_do_not_flush_older_small_backings() {
    let mut pool = FreePool::default();
    pool.release(key(4), Role::Layer, 1);
    pool.release(key(32), Role::Scratch, 2);
    pool.release(key(32), Role::Layer, 3);
    assert_eq!(
        pool.trim_to_bytes(64, 64),
        Trimmed {
            total: 2,
            scratch: 1
        }
    );
    assert_eq!(ids(&pool), [1]);
    // Oversized is relative to the entire budget, not the remaining allowance.
    pool.release(key(8), Role::Scratch, 4);
    assert_eq!(
        pool.trim_to_bytes(24, 64),
        Trimmed {
            total: 2,
            scratch: 1
        }
    );
    assert_eq!(pool.bytes, 0);
    assert_usage(&pool);
}

#[test]
fn fitting_backings_never_expire_and_zero_budget_drops_all() {
    let mut pool = FreePool::default();
    pool.release(key(4), Role::Scratch, 1);
    for _ in 0..1000 {
        assert_eq!(pool.trim_to_bytes(16, 16), Trimmed::default());
    }
    assert_eq!(ids(&pool), [1]);
    assert_eq!(
        pool.trim_to_bytes(0, 0),
        Trimmed {
            total: 1,
            scratch: 1
        }
    );
    assert!(pool.buckets.is_empty());
    assert_usage(&pool);
}

#[test]
fn empty_bucket_capacity_is_reused_within_frame_and_pruned_at_frame_end() {
    let mut pool = FreePool::default();
    pool.release(key(4), Role::Layer, 1);
    let capacity = pool.buckets[&key(4)].entries.capacity();
    assert_eq!(pool.take(key(4)), Some(1));
    assert_eq!(pool.buckets.len(), 1);
    pool.release(key(4), Role::Layer, 1);
    assert_eq!(pool.buckets[&key(4)].entries.capacity(), capacity);
    assert_eq!(pool.take(key(4)), Some(1));
    assert_eq!(pool.trim_to_bytes(16, 16), Trimmed::default());
    assert!(pool.buckets.is_empty());
}

#[test]
fn release_sequence_rebases_without_reordering_live_entries() {
    let mut pool = FreePool {
        next_release: u64::MAX - 2,
        ..FreePool::default()
    };
    pool.release(key(4), Role::Scratch, 1);
    pool.release(key(8), Role::Layer, 2);
    pool.release(key(4), Role::Layer, 3);
    assert_eq!(pool.next_release, 3);
    assert_eq!(
        pool.trim_to_bytes(48, 64),
        Trimmed {
            total: 1,
            scratch: 1
        }
    );
    assert_eq!(ids(&pool), [2, 3]);
    assert_eq!(pool.take(key(4)), Some(3));
    assert_usage(&pool);
}

#[test]
fn deterministic_churn_matches_a_simple_ordered_reference() {
    let mut pool = FreePool::default();
    let mut free: Vec<(Key, Role, usize)> = Vec::new();
    let mut active = Vec::new();
    let mut random = 0xfeed_beef_u64;
    for step in 0..10_000 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        if random.is_multiple_of(3) && !active.is_empty() {
            let index = random as usize % active.len();
            let (key, role, id) = active.swap_remove(index);
            pool.release(key, role, id);
            free.push((key, role, id));
        } else {
            let key = Key {
                format: if random & 8 == 0 {
                    wgpu::TextureFormat::Rgba16Float
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                },
                ..key(1 + (random % 12) as u32)
            };
            let actual = pool.take(key);
            let expected = free
                .iter()
                .rposition(|entry| entry.0 == key)
                .map(|index| free.remove(index).2);
            assert_eq!(actual, expected);
            let role = if random & 16 == 0 {
                Role::Scratch
            } else {
                Role::Layer
            };
            active.push((key, role, actual.unwrap_or(step)));
        }
        if step % 7 == 0 {
            let budget = (random >> 32) % 256;
            let allowance = budget.saturating_sub((random >> 16) % 128);
            let mut expected = Trimmed::default();
            let mut bytes: u64 = free.iter().map(|entry| entry.0.bytes()).sum();
            if bytes > allowance {
                free.retain(|(key, role, _)| {
                    if key.bytes() > budget {
                        bytes -= key.bytes();
                        expected.total += 1;
                        expected.scratch += usize::from(*role == Role::Scratch);
                        false
                    } else {
                        true
                    }
                });
                while bytes > allowance {
                    let (key, role, _) = free.remove(0);
                    bytes -= key.bytes();
                    expected.total += 1;
                    expected.scratch += usize::from(role == Role::Scratch);
                }
            }
            assert_eq!(pool.trim_to_bytes(allowance, budget), expected);
            let mut expected_ids: Vec<_> = free.iter().map(|entry| entry.2).collect();
            expected_ids.sort_unstable();
            assert_eq!(ids(&pool), expected_ids);
        }
        assert_usage(&pool);
    }
}
