# SUGGESTIONS — Ahocorasick NER

Proposals for future enhancements, ranked by impact and effort.

---

## High Impact, Low Effort

### 1. Greedy vs. Maximum Coverage (15 min)
**Goal**: Add option for maximum coverage matching (select all non-overlapping matches, not just longest).

**Implementation**:
- Add `overlap_strategy: str` parameter to `tag()` — "longest" (default) or "max_coverage"
- Max coverage: sort matches by (length desc, position asc), select greedily

**Impact**: Enables use case where multiple shorter entities are better than one long one.

---

## Lower Priority

### 2. Parallel Matching (1-2 hours)
**Goal**: Batch-tag multiple texts in parallel using multiprocessing.

**Implementation**:
- New `tag_batch(texts: List[str]) -> List[List[Dict]]`
- Use ProcessPoolExecutor for CPU-bound work
- Return results in same order as input

**Impact**: 2-3x speedup for batch workloads (e.g., document processing pipelines).

---

## Deferred (Research Phase)

### Fuzzy Matching Integration
**Reason**: Requires tuning of similarity threshold; not clear if worth the latency hit.
**Effort**: 2+ hours
**Deferred until**: User request or performance benchmarks show need.

---

## Related Issues
- See `AUDIT.md` for known limitations
- See `FAQ.md` for feature requests from users
