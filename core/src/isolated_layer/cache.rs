use super::ContentChangeHandle;

/// Eviction priority requested for a cached isolated-layer output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheResidencyPriority {
    /// Is evicted before protected cached values under renderer-owned memory pressure.
    ///
    /// Residency eligibility is driven by renderer keep-alive observations, subject to the
    /// renderer's grace period. Holding a [`super::SurfaceHandle`] does not keep pixels resident.
    Normal,
    /// Is evicted after normal-priority cached values under memory pressure.
    ///
    /// Protected priority is not a residency guarantee. Residency eligibility is driven by
    /// renderer keep-alive observations, subject to the renderer's grace period, and pixels may
    /// still be evicted when the renderer-owned GPU texture budget is exceeded. Holding a
    /// [`super::SurfaceHandle`] does not keep pixels resident.
    Protected,
}

/// A stable content-input identity paired with an observed generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Revision {
    pub(crate) identity: u64,
    pub(crate) generation: u64,
}

impl Revision {
    /// Returns the stable content-input identity.
    pub fn identity(self) -> u64 {
        self.identity
    }

    /// Returns the observed content generation.
    pub fn generation(self) -> u64 {
        self.generation
    }
}

/// A deterministic snapshot of the content inputs observed by a retained-output producer.
///
/// Revisions are sorted by identity and deduplicated. If duplicate observations of one identity
/// disagree on generation, [`is_conflicted`](Self::is_conflicted) is true and the canonical list
/// retains the greatest observed generation. Renderers must fail closed for conflicted stamps.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentStamp {
    pub(crate) revisions: Vec<Revision>,
    pub(crate) conflicted: bool,
}

impl ContentStamp {
    /// Returns the normalized content-input revisions.
    pub fn revisions(&self) -> &[Revision] {
        &self.revisions
    }

    /// Returns whether no content-change evidence was supplied.
    ///
    /// An empty stamp is not sufficient evidence for output reuse. Callers caching intentionally
    /// static content should supply a never-marked [`ContentChangeHandle`].
    pub fn is_empty(&self) -> bool {
        self.revisions.is_empty()
    }

    /// Returns whether duplicate observations disagreed on a generation.
    pub fn is_conflicted(&self) -> bool {
        self.conflicted
    }
}

/// A request to keep one cached isolated-layer output slot resident for the current frame.
///
/// A keep-alive addresses only the output slot and its residency priority. It never creates
/// pixels, updates the slot's stored input signature, or grants permission to reuse its output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKeepAlive {
    pub(crate) identity: u64,
    pub(crate) priority: CacheResidencyPriority,
}

impl CacheKeepAlive {
    /// Returns the stable output-slot identity.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Returns the requested eviction priority.
    pub fn priority(&self) -> CacheResidencyPriority {
        self.priority
    }
}

/// A request to cache the final pre-composite output of an isolated layer.
///
/// The output slot selected by [`identity`](Self::identity) is separate from the caller-managed
/// content revisions in [`stamp`](Self::stamp). The retained output is reusable only when the
/// complete layer input signature matches; this request supplies its content-change component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRequest {
    pub(crate) identity: u64,
    pub(crate) stamp: ContentStamp,
    pub(crate) observed_content: Vec<ContentChangeHandle>,
    pub(crate) priority: CacheResidencyPriority,
}

impl CacheRequest {
    /// Returns the stable output-slot identity.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Returns the requested eviction priority.
    pub fn priority(&self) -> CacheResidencyPriority {
        self.priority
    }

    /// Returns the normalized content revisions observed when this request was created.
    pub fn revisions(&self) -> &[Revision] {
        self.stamp.revisions()
    }

    /// Returns the normalized content stamp observed when this request was created.
    pub fn stamp(&self) -> &ContentStamp {
        &self.stamp
    }

    /// Returns whether this request contains at least one content-change observation.
    ///
    /// Renderers must bypass retained-output lookup and storage when this returns `false`.
    pub fn has_content_evidence(&self) -> bool {
        !self.stamp.is_empty()
    }

    /// Returns whether duplicate observations disagreed on a generation.
    pub fn is_conflicted(&self) -> bool {
        self.stamp.is_conflicted()
    }

    /// Observes the current generations of the same normalized content handles.
    ///
    /// Renderers can use this at store time to ensure the content evidence did not change after
    /// lookup and while the output was being rendered.
    pub fn current_stamp(&self) -> ContentStamp {
        super::content_stamp(self.observed_content.iter())
    }

    /// Returns whether this request still has sufficient, unconflicted, unchanged evidence.
    ///
    /// Empty evidence fails closed. A caller caching intentionally static content should observe
    /// a never-marked [`ContentChangeHandle`].
    pub fn is_current(&self) -> bool {
        self.has_content_evidence()
            && !self.stamp.is_conflicted()
            && self.current_stamp() == self.stamp
    }

    /// Creates a request for the same output slot from the currently observed generations.
    ///
    /// This is useful when a renderer needs the latest evidence after rejecting a stale lease.
    pub fn resnapshot(&self) -> Self {
        Self {
            identity: self.identity,
            stamp: self.current_stamp(),
            observed_content: self.observed_content.clone(),
            priority: self.priority,
        }
    }
}
