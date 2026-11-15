# Portability changes relative to the archived analysis snapshot

- Source-directory and interpreter constants were replaced by repository-relative paths and the current Python executable.
- Fresh outputs are isolated under CA_RUN_ROOT; raw aggregate inputs and archived published outputs remain read-only.
- Exact original JSON configuration bytes were restored. The earlier delivery ZIP had reserialized JSON, which preserved content but broke historical config-byte hashes.
- Config, raw input and pinned generating-snapshot checks remain active. Code hashes from the original machine are retained as provenance; portable IO wrappers are checked by the current MANIFEST instead of being required to equal obsolete original code hashes.
- The sensitivity report compares scientific estimates/bounds using its original relative tolerance. Only four named near-zero numerical diagnostic fields (LP/vertex residuals and constraint/remainder violations) additionally permit absolute differences up to 1e-10. This avoids treating 0 versus 1e-12 rounding noise as a failed scientific reproduction. No estimates, interval endpoints, generating laws or archived outputs were changed.
- Git attributes disable newline conversion, including for immutable source snapshots. This preserves byte-level input hashes across Git clones and operating systems.

- Non-finite recomputed scientific values cannot pass comparison against finite archived values.
- Output roots that contain the repository, or lie in its tracked directories, are rejected. Custom output roots must be outside the repository or under outputs/.
