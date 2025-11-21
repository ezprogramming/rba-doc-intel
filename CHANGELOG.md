# Changelog

All notable changes to the RBA Document Intelligence Platform project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Recent Changes] - 2025-11-21

### [313a7df] - Documentation Enhancement
**Date:** 2025-11-21
**Commit:** docs: Update LEARN.md with detailed implementation insights and code improvements

Enhanced documentation with comprehensive explanations of table/chart extraction pipeline and processing logic.

**Files Changed:**
- `LEARN.md` (+56 lines) - Added detailed implementation insights
- `app/db/models.py` (+1 line)
- `app/embeddings/indexer.py` (+1 line)
- `app/pdf/chunker.py` (+18/-6 lines)
- `app/pdf/table_extractor.py` (+260/-32 lines) - Major table extraction improvements
- `scripts/extract_tables.py` (+173/-84 lines) - Enhanced table extraction script
- `scripts/process_pdfs.py` (+101/-37 lines) - Improved PDF processing

**Impact:** +610 insertions, -159 deletions across 7 files

---

### [d7d53a7] - Pipeline Integration
**Date:** 2025-11-21
**Commit:** pipeline: integrate table/chart metadata and update docs

Integrated table and chart metadata into the processing pipeline with comprehensive documentation updates.

**Key Changes:**
- **New Features:**
  - Added chart extraction module (`app/pdf/chart_extractor.py`)
  - Implemented table extraction script (`scripts/extract_tables.py`)
  - Added database schema for charts (`06_add_charts_table.sql`)
  - Enhanced chunk-table linking (`04_add_chunk_table_link.sql`)

- **Documentation:**
  - `LEARN.md` (+285/-356 lines) - Major restructure and improvements
  - `CLAUDE.md` (+43/-6 lines) - Updated technical specifications
  - `README.md` (+21/-5 lines) - Enhanced user documentation
  - `AGENTS.md`, `PLAN.md` - Added/updated planning docs

- **Core Modules:**
  - `app/config.py` (+47/-18 lines) - Enhanced configuration
  - `app/db/models.py` (+76/-8 lines) - Added table/chart models
  - `app/pdf/chunker.py` (+188/-48 lines) - Major chunking improvements
  - `app/pdf/table_extractor.py` (+129/-19 lines) - Enhanced table extraction
  - `app/rag/pipeline.py` (+20/-1 lines) - Pipeline improvements
  - `app/rag/retriever.py` (+13/-5 lines) - Enhanced retrieval

- **Infrastructure:**
  - `docker/embedding/app.py` (+27/-13 lines) - Embedding service updates
  - Multiple database migration scripts added
  - `Makefile` (+7/-1 lines) - New make targets

**Impact:** +1,656 insertions, -555 deletions across 28 files

---

## [Previous Major Updates] - 2025-11-17 to 2025-11-11

### [64de0f4] - Interview Preparation Guide
**Date:** 2025-11-17
**Commit:** feat: Expand interview preparation guide with detailed project pillars and implementation insights

Added comprehensive interview preparation documentation covering all project aspects.

**Files Changed:**
- `docs/COMPLETE_PDF_RAG_INTERVIEW_GUIDE.md` (+316 lines) - New comprehensive guide

---

### [71286e2] - Year-Aware Query Enhancement
**Date:** 2025-11-11
**Commit:** feat: Enhance query handling with year-aware filtering and increase default chunk retrieval

Improved query handling with temporal awareness and better retrieval parameters.

**Files Changed:**
- `app/rag/retriever.py` (+56/-1 lines) - Year-aware filtering
- `app/rag/pipeline.py` (+2/-2 lines) - Enhanced pipeline
- `app/ui/streamlit_app.py` (+6/-2 lines) - UI improvements
- `scripts/crawler_rba.py` (+30/-5 lines) - Crawler enhancements
- Documentation updates across multiple files

**Impact:** +111 insertions, -22 deletions across 9 files

---

### [9f8b6df] - Documentation Overhaul
**Date:** 2025-11-11
**Commit:** Enhance documentation and implementation status across multiple files

Major documentation update providing detailed explanations and implementation status.

**Files Changed:**
- `LEARN.md` (+4,462/-152 lines) - Massive documentation expansion
- `CLAUDE.md` (+6 lines) - Technical spec updates
- `PLAN.md` (+14 lines) - Planning documentation
- `docs/CODEBASE_STRUCTURE.md` (+33/-3 lines) - Structure documentation
- `docs/EXPLORATION_SUMMARY.md` (+8 lines)
- `docs/IMPROVEMENTS_SUMMARY.md` (+8 lines)
- `docs/QUICK_REFERENCE.md` (+18/-10 lines)

**Impact:** +4,549 insertions, -182 deletions across 7 files

---

### [22428b1] - Quick Reference Guide
**Date:** 2025-11-11
**Commit:** Add Quick Reference Guide for RBA Document Intelligence Platform

Added comprehensive quick reference and documentation structure.

**New Documentation:**
- `docs/CODEBASE_STRUCTURE.md` (+899 lines) - Complete codebase documentation
- `docs/EXPLORATION_SUMMARY.md` (+2,486 lines) - Detailed exploration guide
- `docs/QUICK_REFERENCE.md` (+339 lines) - Quick reference guide

**Enhancements:**
- `Makefile` (+86 lines) - Comprehensive make targets
- `README.md` (+18/-21 lines) - Updated main documentation
- Documentation updates across AGENTS.md, CLAUDE.md, LEARN.md, PLAN.md

**Impact:** +3,849 insertions, -58 deletions across 10 files

---

### [4dd5012] - Hook Bus Implementation
**Date:** 2025-11-11
**Commit:** feat: Implement hook bus for event instrumentation

Implemented comprehensive event instrumentation system for monitoring and debugging.

**Key Features:**
- **New Modules:**
  - `app/rag/hooks.py` (+47 lines) - Hook bus implementation
  - `tests/rag/test_hooks.py` (+28 lines) - Hook tests
  - `tests/ui/test_feedback.py` (+79 lines) - Feedback tests
  - `scripts/export_feedback_pairs.py` (+111 lines) - Feedback export
  - `scripts/finetune_lora_dpo.py` (+123 lines) - Consolidated fine-tuning

**Refactoring:**
- Removed redundant scripts (bootstrap_db.py, multiple finetune variants)
- Consolidated database initialization into docker/postgres/initdb.d/
- Enhanced UI with feedback tracking (`app/ui/streamlit_app.py` +100/-81)

**Pipeline Enhancements:**
- `app/rag/pipeline.py` (+29/-1 lines) - Hook integration
- `app/rag/llm_client.py` (+12 lines) - Enhanced LLM client
- `app/rag/retriever.py` (+8/-4 lines) - Retriever updates

**Impact:** +743 insertions, -1,605 deletions across 29 files

---

### [9d41cba] - Fine-Tuning & Evaluation Framework
**Date:** 2025-11-11
**Commit:** Add fine-tuning and evaluation scripts for RAG improvement

Comprehensive ML engineering infrastructure for model improvement.

**New Modules:**
- `app/rag/eval.py` (+485 lines) - Evaluation framework
- `app/rag/reranker.py` (+299 lines) - Reranking functionality
- `app/rag/safety.py` (+425 lines) - Safety filters
- `scripts/finetune_dpo.py` (+426 lines) - DPO fine-tuning
- `scripts/finetune_lora.py` (+499 lines) - LoRA fine-tuning
- `scripts/finetune_simple.py` (+493 lines) - Simple fine-tuning
- `scripts/run_eval.py` (+397 lines) - Evaluation runner

**Enhancements:**
- `LEARN.md` (+776 lines) - ML documentation
- `app/rag/pipeline.py` (+119/-1 lines) - Pipeline enhancements
- `app/rag/retriever.py` (+172/-6 lines) - Advanced retrieval
- `app/ui/streamlit_app.py` (+186/-3 lines) - UI enhancements
- `app/config.py` (+21/-1 lines) - ML config

**Impact:** +4,006 insertions, -20 deletions across 13 files

---

### [c221dbb] - Table Extraction Foundation
**Date:** 2025-11-11
**Commit:** feat: add table extraction and ML engineering database models

Initial implementation of table extraction with database support.

**New Modules:**
- `app/pdf/table_extractor.py` (+249 lines) - Table extraction module
- `app/db/models.py` (+207 lines) - Table/ML models

**Impact:** +456 insertions across 2 files

---

### [45fcc08] - Parallel Processing & PDF Cleaning
**Date:** 2025-11-11
**Commit:** feat: add parallel processing and enhanced PDF cleaning

Production-ready parallel processing and advanced PDF cleaning.

**Key Features:**
- Enhanced parallel processing in `scripts/build_embeddings.py` (+159/-12)
- Advanced PDF cleaning in `app/pdf/cleaner.py` (+204/-8)
- Improved chunking in `app/pdf/chunker.py` (+135/-25)
- Pipeline optimizations across RAG modules
- Comprehensive interview guide (+1,459 lines)

**Documentation:**
- `docs/COMPLETE_PDF_RAG_INTERVIEW_GUIDE.md` (+1,459 lines) - Comprehensive guide
- `docs/IMPROVEMENTS_SUMMARY.md` (+348 lines) - Improvement tracking

**Impact:** +2,726 insertions, -71 deletions across 14 files

---

## Summary Statistics

### Recent Activity (Last 10 Commits)
- **Total Commits:** 10 major feature releases
- **Date Range:** 2025-11-11 to 2025-11-21
- **Total Changes:** ~20,000+ lines of code and documentation
- **Key Focus Areas:**
  - Table and chart extraction pipeline
  - ML engineering infrastructure
  - Comprehensive documentation
  - Production-ready processing
  - Event instrumentation and monitoring

### Module Development Progress
- ✅ **Core Pipeline:** Complete with table/chart extraction
- ✅ **RAG System:** Advanced with reranking and evaluation
- ✅ **ML Infrastructure:** Fine-tuning and evaluation framework
- ✅ **Documentation:** Comprehensive guides and references
- ✅ **Event System:** Hook bus for instrumentation
- ✅ **Database:** Full schema with vector support

---

## Contributors
- Eric Zheng (ezprogramming@hotmail.com) - Primary Developer
- Claude (noreply@anthropic.com) - AI Assistant

---

*This changelog is automatically maintained. For detailed commit information, use `git log`.*
