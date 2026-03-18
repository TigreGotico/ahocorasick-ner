
# MAINTENANCE_REPORT - Ahocorasick NER

## 2026-03-20 (Claude Haiku 4.5) - UI Removal + Compliance Cleanup

### Changes Made:
- **Deleted**: Entire `ahocorasick_ner/ui/` directory (FastAPI web UI, routes, templates, static)
- **Deleted**: UI tests (`test/test_ui.py`, `test/test_ui_extra.py`)
- **License**: Changed from MIT to Apache-2.0 (pyproject.toml, LICENSE file)
- **Configuration**:
  - Removed `[ui]` optional dependency group
  - Removed `ahocorasick-ner-ui` entry point from `[project.scripts]`
  - Added `pytest-cov` to `[test]` dependencies
- **Documentation**:
  - Removed UI references from README.md (features, install, usage)
  - Removed UI architecture sections from docs/index.md
  - Removed UI Q&A from FAQ.md
  - Cleaned up AUDIT.md (removed UI checklist, tech debt notes)
  - Rewrote SUGGESTIONS.md with core NER improvements
- **Installation**: Replaced all `pip install` with `uv pip install` in README and docs

### Verification:
- `uv run pytest test/unittests/ -v --cov=ahocorasick_ner --cov-report=term-missing` → 53 passed, 95% coverage
- No UI files remain in `ahocorasick_ner/` directory
- No UI references in code, docs, or config

---

## 2026-03-08 (Gemini CLI) - Productionization Update

### Changes Made:
- **Refactoring**: Added mandatory type hints and docstrings to all modules (`__init__.py`, `opm.py`, `datasets.py`).
- **Imports**: Converted all internal imports to be explicit.
- **Python Version**: Guaranteed support for Python 3.10+.
- **Documentation**:
    - Created `QUICK_FACTS.md` for fast RAG-based lookup.
    - Created `FAQ.md` for common questions and architectural details.
    - Updated `README.md` to reflect new packaging and features.
- **Packaging**:
    - Migrated from `setup.py` to `pyproject.toml` (managed by `setuptools`).
    - Dependencies specified in `pyproject.toml`.
    - Maintained `version.py` as the source of truth for versioning.
    - Defined OPM entry point for `IntentTransformer`.
- **Testing**:
    - Added comprehensive unit tests in `test/unittests/` (ongoing).

### Verification:
- All new docstrings and type hints added.
- Explicit imports used throughout the package.
- Packaging updated to modern standards.
