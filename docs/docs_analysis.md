# Documentation Notes

## What This Folder Covers

This docs folder is now organized around the current codebase instead of the earlier prototype.

Primary files:

- [PROJECT_OVERVIEW.md](c:\Users\Musharraf\Documents\POC\docs\PROJECT_OVERVIEW.md)
- [BACKEND_SPEC.md](c:\Users\Musharraf\Documents\POC\docs\BACKEND_SPEC.md)
- [FRONTEND_SPEC.md](c:\Users\Musharraf\Documents\POC\docs\FRONTEND_SPEC.md)
- [backend_status.md](c:\Users\Musharraf\Documents\POC\docs\backend_status.md)

Reference asset retained:

- `Model IDs Reference Guide.pdf`

## Documentation Intent

### Project overview

Use [PROJECT_OVERVIEW.md](c:\Users\Musharraf\Documents\POC\docs\PROJECT_OVERVIEW.md) when you need:

- product-level understanding
- workflow boundaries
- system architecture at a glance

### Backend spec

Use [BACKEND_SPEC.md](c:\Users\Musharraf\Documents\POC\docs\BACKEND_SPEC.md) when you need:

- API routes
- service responsibilities
- environment variables
- persistence model
- recommendation logic summary

### Frontend spec

Use [FRONTEND_SPEC.md](c:\Users\Musharraf\Documents\POC\docs\FRONTEND_SPEC.md) when you need:

- page responsibilities
- component roles
- client-side API usage
- user interaction flow

### Backend status

Use [backend_status.md](c:\Users\Musharraf\Documents\POC\docs\backend_status.md) when you need:

- current capability snapshot
- constraints
- practical risk areas
- likely next improvements

## Source of Truth

These docs were aligned to the current implementation in:

- `backend/`
- `frontend/`
- `model_training/` fallback artifacts and recommendation support files

If behavior changes in code, the source of truth is still the implementation first.

## Known Gaps Still Outside This Folder

- There is no formal schema migration document for Supabase
- There is no deployment guide yet
- There is no dedicated troubleshooting runbook yet
- There is no architecture decision record for the evaluator pool design
