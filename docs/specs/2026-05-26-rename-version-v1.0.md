# Rename Project Version to v1.0

**Date**: 2026-05-26
**Status**: approved

## Scope

Replace all project version identifiers (`v6.2`, `v2.3`, `v6.0`, `v2.2`, `6.2.0`) with v1.0 equivalents across 6 files, 11 occurrences.

## Changes

| File | Current | To |
|------|---------|-----|
| `pyproject.toml:7` | `version = "6.2.0"` | `version = "1.0.0"` |
| `README.md:1` | `# 舞蹈评分系统 v6.2` | `# 舞蹈评分系统 v1.0` |
| `README.md:5` | `## v6.2 更新内容` | `## v1.0 更新内容` |
| `README.md:7` | `（v6.0 → v6.2）` | `（v1.0）` |
| `README.md:15` | `（v2.2 → v2.3）` | `（v1.0）` |
| `scripts/score.py:30` | `舞蹈评分 v6.2` | `舞蹈评分 v1.0` |
| `scripts/split.py:28` | `8拍慢动作分段 v2.3` | `8拍慢动作分段 v1.0` |
| `src/dance_scoring/gui/app.py:19,46` | `舞蹈评分系统 v6.2` | `舞蹈评分系统 v1.0` |
| `docs/gui-requirements.md:5,29` | `v6.2`, `v2.3` | `v1.0` |

## Exclusions

- `check_env.py` — `mp.__version__` is MediaPipe library version, not project version
- `docs/specs/2026-05-26-project-restructure-design.md` — historical design record, kept as-is

## Commit

`chore: rename version to v1.0`
