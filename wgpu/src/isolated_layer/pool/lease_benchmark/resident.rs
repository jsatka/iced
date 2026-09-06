//! Checked pool-resident prototype. Intentionally test-only: short borrows here
//! do not resolve the renderer's recursive text-viewport borrow requirements.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

const PAGE: usize = 64;
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

fn next_owner() -> u64 {
    let mut current = NEXT_OWNER.load(Ordering::Relaxed);
    loop {
        let next = current.checked_add(1).expect("pool identity exhausted");
        match NEXT_OWNER.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return current,
            Err(actual) => current = actual,
        }
    }
}

// Not Clone/Copy: normal callers cannot duplicate a lease. Generation and owner
// validation are retained in release builds, not just debug assertions.
pub(super) struct Token {
    owner: u64,
    index: u32,
    generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Vacant,
    Free,
    Leased,
}

struct Slot<T> {
    generation: u32,
    state: State,
    value: Option<T>,
}

struct Slots<T> {
    owner: u64,
    pages: Vec<Box<[Slot<T>; PAGE]>>,
    vacant: Vec<u32>,
}

impl<T> Default for Slots<T> {
    fn default() -> Self {
        Self {
            owner: next_owner(),
            pages: Vec::new(),
            vacant: Vec::new(),
        }
    }
}

impl<T> Slots<T> {
    fn slot(&self, token: &Token, expected: State) -> &Slot<T> {
        assert_eq!(token.owner, self.owner, "wrong pool");
        let slot = &self.pages[token.index as usize / PAGE][token.index as usize % PAGE];
        assert_eq!(token.generation, slot.generation, "stale token");
        assert_eq!(slot.state, expected, "wrong lease state");
        slot
    }

    fn slot_mut(&mut self, token: &Token, expected: State) -> &mut Slot<T> {
        assert_eq!(token.owner, self.owner, "wrong pool");
        let slot = &mut self.pages[token.index as usize / PAGE][token.index as usize % PAGE];
        assert_eq!(token.generation, slot.generation, "stale token");
        assert_eq!(slot.state, expected, "wrong lease state");
        slot
    }

    fn insert(&mut self, target: T) -> Token {
        if self.vacant.is_empty() {
            let start = self.pages.len() * PAGE;
            let last = u32::try_from(start + PAGE - 1).expect("slot index exhausted");
            self.pages.push(Box::new(std::array::from_fn(|_| Slot {
                generation: 0,
                state: State::Vacant,
                value: None,
            })));
            self.vacant.extend((start as u32..=last).rev());
        }
        let index = self.vacant.pop().unwrap();
        let slot = &mut self.pages[index as usize / PAGE][index as usize % PAGE];
        assert_eq!(slot.state, State::Vacant);
        slot.generation = slot
            .generation
            .checked_add(1)
            .expect("exhausted slot must not be reused");
        slot.state = State::Leased;
        slot.value = Some(target);
        Token {
            owner: self.owner,
            index,
            generation: slot.generation,
        }
    }

    fn remove(&mut self, token: Token, expected: State) {
        let slot = self.slot_mut(&token, expected);
        let value = slot.value.take().unwrap();
        slot.state = State::Vacant;
        if slot.generation != u32::MAX {
            self.vacant.push(token.index);
        }
        drop(value);
    }

    fn heap_bytes(&self) -> usize {
        self.pages.capacity() * std::mem::size_of::<Box<[Slot<T>; PAGE]>>()
            + self.pages.len() * std::mem::size_of::<[Slot<T>; PAGE]>()
            + self.vacant.capacity() * std::mem::size_of::<u32>()
    }
}

pub(super) struct Resident<T> {
    free: FreePool<Token>,
    slots: Slots<T>,
}

impl<T> Default for Resident<T> {
    fn default() -> Self {
        Self {
            free: FreePool::default(),
            slots: Slots::default(),
        }
    }
}

impl<T: Resource> Storage<T> for Resident<T> {
    type Lease = Token;
    fn take(&mut self, key: Key) -> Option<Token> {
        let token = self.free.take(key)?;
        self.slots.slot_mut(&token, State::Free).state = State::Leased;
        Some(token)
    }
    fn insert(&mut self, target: T) -> Token {
        self.slots.insert(target)
    }
    fn get<'a>(&'a self, token: &'a Token) -> &'a T {
        self.slots
            .slot(token, State::Leased)
            .value
            .as_ref()
            .unwrap()
    }
    fn get_mut<'a>(&'a mut self, token: &'a mut Token) -> &'a mut T {
        self.slots
            .slot_mut(token, State::Leased)
            .value
            .as_mut()
            .unwrap()
    }
    fn release(&mut self, token: Token) {
        let slot = self.slots.slot_mut(&token, State::Leased);
        let target = slot.value.as_ref().unwrap();
        let (key, role) = (target.key(), target.role());
        slot.state = State::Free;
        self.free.release(key, role, token);
    }
    fn trim(&mut self, allowance: u64, budget: u64) -> usize {
        let Self { free, slots } = self;
        free.trim_to_bytes_with(allowance, budget, |token| slots.remove(token, State::Free))
            .total
    }
    fn free(&self) -> (u64, u64) {
        (self.free.bytes, self.free.scratch_bytes)
    }
    fn heap_bytes(&self, _: usize) -> usize {
        free_heap(&self.free) + self.slots.heap_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(id: u64) -> CpuTarget {
        CpuTarget {
            key: Key::new(wgpu::TextureFormat::Rgba8Unorm, Size::new(64, 64)),
            role: Role::Layer,
            id,
            payload: [id; 27],
        }
    }

    #[test]
    fn resident_growth_preserves_addresses_and_active_exclusion() {
        let mut pool = Resident::default();
        let first = pool.insert(resource(1));
        let key = pool.get(&first).key;
        let address = std::ptr::from_ref(pool.get(&first));
        let others: Vec<_> = (2..200).map(|id| pool.insert(resource(id))).collect();
        assert_eq!(address, std::ptr::from_ref(pool.get(&first)));
        assert!(pool.take(key).is_none());
        assert_eq!(pool.trim(0, 0), 0);
        pool.release(first);
        let again = pool.take(key).unwrap();
        assert_eq!(address, std::ptr::from_ref(pool.get(&again)));
        for token in others {
            pool.release(token);
        }
        pool.release(again);
        assert_eq!(pool.trim(0, 0), 199);
        assert_eq!(pool.free(), (0, 0));
        assert!(
            pool.slots
                .pages
                .iter()
                .flat_map(|page| page.iter())
                .all(|slot| slot.value.is_none())
        );
    }

    #[test]
    fn stale_wrong_pool_and_released_tokens_are_rejected() {
        let mut pool = Resident::default();
        let token = pool.insert(resource(1));
        let duplicate = Token {
            owner: token.owner,
            index: token.index,
            generation: token.generation,
        };
        let mut other = Resident::default();
        let _other_token = other.insert(resource(2));
        assert!(std::panic::catch_unwind(|| other.get(&duplicate)).is_err());
        pool.release(token);
        assert!(std::panic::catch_unwind(|| pool.get(&duplicate)).is_err());
        assert_eq!(pool.trim(0, 0), 1);
        let replacement = pool.insert(resource(3));
        assert_eq!(duplicate.index, replacement.index);
        assert_ne!(duplicate.generation, replacement.generation);
        assert!(std::panic::catch_unwind(|| pool.get(&duplicate)).is_err());
        assert_eq!(pool.get(&replacement).id, 3);
    }

    #[test]
    fn generation_exhaustion_retires_the_slot() {
        let mut pool = Resident::default();
        let mut token = pool.insert(resource(1));
        pool.slots.slot_mut(&token, State::Leased).generation = u32::MAX;
        token.generation = u32::MAX;
        let old_index = token.index;
        pool.release(token);
        assert_eq!(pool.trim(0, 0), 1);
        let next = pool.insert(resource(2));
        assert_ne!(old_index, next.index);
    }

    #[test]
    fn boxed_growth_and_returns_preserve_addresses_without_aliasing_leases() {
        let mut pool = Boxed::default();
        let first = pool.insert(resource(1));
        let key = first.key;
        let address = std::ptr::from_ref(pool.get(&first));
        let others: Vec<_> = (2..200).map(|id| pool.insert(resource(id))).collect();
        assert!(pool.take(key).is_none());
        assert_eq!(pool.trim(0, 0), 0);
        pool.release(first);
        let again = pool.take(key).unwrap();
        assert_eq!(address, std::ptr::from_ref(pool.get(&again)));
        for target in others {
            pool.release(target);
        }
        pool.release(again);
        assert_eq!(pool.trim(0, 0), 199);
    }
}
