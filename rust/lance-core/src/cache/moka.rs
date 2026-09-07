// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::Future;

use crate::Result;

use super::CacheCodec;
use super::backend::{CacheBackend, CacheEntry, InternalCacheKey};

/// Internal record stored in the moka cache.
#[derive(Clone, Debug)]
struct MokaCacheEntry {
    entry: CacheEntry,
    size_bytes: usize,
}

/// Default [`CacheBackend`] backed by a [moka](https://crates.io/crates/moka) cache.
///
/// Nonzero capacities provide weighted eviction and concurrent-load deduplication.
/// A zero capacity invokes each loader independently without caching.
pub struct MokaCacheBackend {
    cache: moka::future::Cache<InternalCacheKey, MokaCacheEntry>,
}

impl std::fmt::Debug for MokaCacheBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MokaCacheBackend")
            .field("entry_count", &self.cache.entry_count())
            .finish()
    }
}

impl MokaCacheBackend {
    pub fn with_capacity(capacity: usize) -> Self {
        let cache = moka::future::Cache::builder()
            .max_capacity(capacity as u64)
            .weigher(|_, v: &MokaCacheEntry| v.size_bytes.try_into().unwrap_or(u32::MAX))
            .support_invalidation_closures()
            .build();
        Self { cache }
    }

    pub fn no_cache() -> Self {
        Self {
            cache: moka::future::Cache::new(0),
        }
    }
}

#[async_trait]
impl CacheBackend for MokaCacheBackend {
    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.cache.get(key).await.map(|r| r.entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
    ) {
        self.cache
            .insert(key.clone(), MokaCacheEntry { entry, size_bytes })
            .await;
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        _codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        // Avoid Moka's single-flight waiters when nothing can be cached.
        if self.cache.policy().max_capacity() == Some(0) {
            return loader.await.map(|(entry, _)| (entry, false));
        }

        // Use moka's built-in dedup: optionally_get_with runs the init future
        // at most once per key, even under concurrent access.
        let (error_tx, error_rx) = tokio::sync::oneshot::channel();

        // Track whether the loader actually ran (= cache miss).
        let was_miss = Arc::new(AtomicBool::new(false));
        let was_miss_clone = was_miss.clone();

        let init = async move {
            was_miss_clone.store(true, Ordering::Relaxed);
            match loader.await {
                Ok((entry, size_bytes)) => Some(MokaCacheEntry { entry, size_bytes }),
                Err(e) => {
                    let _ = error_tx.send(e);
                    None
                }
            }
        };

        let owned_key = key.clone();
        match self.cache.optionally_get_with(owned_key, init).await {
            Some(record) => {
                let was_cached = !was_miss.load(Ordering::Relaxed);
                Ok((record.entry, was_cached))
            }
            None => match error_rx.await {
                Ok(err) => Err(err),
                Err(_) => Err(crate::Error::internal(
                    "Failed to retrieve error from cache loader",
                )),
            },
        }
    }

    async fn invalidate_prefix(&self, prefix: &str) {
        let prefix = prefix.to_owned();
        self.cache
            .invalidate_entries_if(move |key, _value| key.starts_with(&prefix))
            .expect("Cache configured correctly");
    }

    async fn clear(&self) {
        self.cache.invalidate_all();
        self.cache.run_pending_tasks().await;
    }

    async fn num_entries(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.cache.entry_count() as usize
    }

    async fn size_bytes(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.cache.weighted_size() as usize
    }

    fn approx_num_entries(&self) -> usize {
        self.cache.entry_count() as usize
    }

    fn approx_size_bytes(&self) -> usize {
        // Iterate rather than using `weighted_size()` because moka's
        // weighted_size can be stale without `run_pending_tasks()`, which
        // is async and can't be called from this synchronous context.
        self.cache.iter().map(|(_, v)| v.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;

    use futures::{FutureExt, poll};
    use tokio::sync::oneshot;

    use super::*;

    #[rstest::rstest]
    #[case::zero_capacity_success(MokaCacheBackend::with_capacity(0), false)]
    #[case::zero_capacity_error(MokaCacheBackend::with_capacity(0), true)]
    #[case::no_cache_success(MokaCacheBackend::no_cache(), false)]
    #[case::no_cache_error(MokaCacheBackend::no_cache(), true)]
    #[tokio::test]
    async fn zero_capacity_loaders_run_independently(
        #[case] backend: MokaCacheBackend,
        #[case] loader_fails: bool,
    ) {
        let key = InternalCacheKey::new("dataset".into(), "file".into(), "metadata");
        let first_entry: CacheEntry = Arc::new(1_u64);
        let second_entry: CacheEntry = Arc::new(2_u64);
        let (release_tx, release_rx) = oneshot::channel();
        let entry = first_entry.clone();
        let mut first = pin!(backend.get_or_insert(
            &key,
            Box::pin(async move {
                release_rx.await.unwrap();
                Ok((entry, 8))
            }),
            None,
        ));
        assert!(poll!(first.as_mut()).is_pending());

        let entry = second_entry.clone();
        let second = backend
            .get_or_insert(
                &key,
                Box::pin(async move {
                    if loader_fails {
                        Err(crate::Error::invalid_input("independent loader failed"))
                    } else {
                        Ok((entry, 8))
                    }
                }),
                None,
            )
            .now_or_never()
            .expect("a disabled cache must not wait for another loader on the same key");
        if loader_fails {
            let error = second.unwrap_err();
            assert!(matches!(&error, crate::Error::InvalidInput { .. }));
            assert!(error.to_string().contains("independent loader failed"));
        } else {
            let (entry, was_cached) = second.unwrap();
            assert!(Arc::ptr_eq(&entry, &second_entry));
            assert!(!was_cached);
        }

        release_tx.send(()).unwrap();
        let (entry, was_cached) = first.await.unwrap();
        assert!(Arc::ptr_eq(&entry, &first_entry));
        assert!(!was_cached);
        assert!(backend.get(&key, None).await.is_none());
        assert_eq!(backend.num_entries().await, 0);
        assert_eq!(backend.size_bytes().await, 0);
    }
}
