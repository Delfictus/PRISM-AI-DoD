# Legacy Folders - Documentation Moved

**Status**: LEGACY - Content moved to `docs/` structure
**Date**: 2025-10-17

## What Happened

All documentation from numbered folders has been successfully moved to the centralized `docs/` structure:

## Folder Migration Map

| Legacy Folder | New Location | Status |
|---------------|--------------|---------|
| `00-Constitution/` | `docs/governance/` | ✅ Moved |
| `01-Governance-Engine/` | `docs/governance/progress-tracking/` | ✅ Moved |
| `01-Rapid-Implementation/` | `docs/development/rapid-implementation/` | ✅ Moved |
| `02-Documentation/` | `docs/development/` & `docs/governance/compliance/` | ✅ Moved |
| `03-Code-Templates/` | `docs/development/code-templates/` | ✅ Moved |
| `06-Plans/` | `docs/plans/` | ✅ Moved |
| `07-Web-Platform/` | `docs/architecture/missions/web-platform/` | ✅ Moved |
| `08-Mission-Charlie-LLM/` | `docs/architecture/missions/charlie-llm/` | ✅ Moved |

## New Documentation Structure

All documentation is now organized under `docs/`:
- `docs/getting-started/` - Quick start guides
- `docs/architecture/` - System design & missions
- `docs/governance/` - Constitutional enforcement & progress tracking
- `docs/development/` - Contributing, testing, rapid implementation, code templates
- `docs/plans/` - Strategic planning
- `docs/obsidian-vault/` - Personal vault & governance engine

## Cleanup Recommendation

These legacy folders can be safely removed once you verify all content is accessible in the new `docs/` structure:

```bash
# When ready to clean up:
rm -rf 00-Constitution/ 01-Governance-Engine/ 01-Rapid-Implementation/
rm -rf 02-Documentation/ 03-Code-Templates/ 06-Plans/
rm -rf 07-Web-Platform/ 08-Mission-Charlie-LLM/
```

## Verification

To verify all content was moved successfully:
```bash
# Check docs structure
find docs/ -name "*.md" | wc -l  # Should be 69+ files

# Check main documentation hub
cat docs/index.md
```

All references in documentation have been updated to point to the new locations.