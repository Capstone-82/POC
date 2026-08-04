# Routing Improvements Summary

## Objective

These changes improve model selection for production use. The previous router primarily preferred the cheapest eligible model. The updated approach prioritizes quality, capability fit, safety, and enterprise requirements before considering cost.

## Files Modified

| File | Purpose |
|---|---|
| `prompt_profiling&model_routing/model_registry_v3.json` | Adds verified model metadata, lifecycle information, provider metadata, capability data, and weighted-routing policy settings. |
| `prompt_profiling&model_routing/router.py` | Replaces cost-first ranking with feasibility-gated weighted routing. |
| `prompt_profiling&model_routing/routing_models.py` | Adds optional model metadata and explainable routing-score fields. |
| `backend/routers/profiling.py` | Adds optional enterprise criticality and required-capability inputs to the existing routing API. |

## Major Improvements

### Quality-over-cost routing

Cost is no longer the primary sorting rule. A model must first satisfy quality and capability expectations; cost is used only as one factor among otherwise suitable choices. Critical T3 requests assign no weight to cost.

### Weighted scoring

Candidates receive an explainable score based on quality, capability fit, context headroom, reliability, latency, and cost efficiency. This replaces single-rule or cost-first decisions.

### Tier-specific routing

T1, T2, and T3 requests use different score weights. T1 retains greater cost and latency sensitivity, while T3 emphasizes quality and capability fit.

### Enterprise criticality

The routing API accepts an optional enterprise-criticality value. High, critical, regulated, and safety-critical T3 requests use a stricter quality profile and remove cost influence.

### Feasibility gates

Models are excluded before scoring when they are deprecated, unavailable, manually restricted, too small for the requested output or full context, or missing a required capability.

### Context handling

The router now validates input with a safety margin, requested output size, and total request context. It also favors candidates with more remaining context headroom.

### Capability matching

Routing incorporates domain, task type, intent, reasoning requirements, research dependency, and context needs. Callers may also provide explicit required capabilities.

### Lifecycle and deprecated-model handling

Registry lifecycle status is now authoritative. Deprecated, retired, and disabled models are excluded from new routing decisions. Verified replacement guidance is recorded where available.

### Explainable routing decisions

Each recommendation includes an overall routing score and a factor-level score breakdown. This makes it easier to understand why a model was selected or rejected.

### Fallback recommendations

The recommendation list is now an ordered set of feasible fallback candidates. Near ties are resolved deterministically using quality, reliability, context, latency, cost, and stable model identity.

### Registry improvements

The registry now supports schema versioning, provider metadata, lifecycle status, canonical API IDs, verification status, capability tags, and total-context information. Facts that are not explicitly verified default to `Needs Manual Verification`.

## Why These Changes Were Needed

The previous implementation filtered candidates by tier and basic input capacity, then generally ranked them by estimated cost. This could select a cheaper model for a complex T3 request even when another eligible model had a stronger quality or capability fit. It also lacked output and total-context validation, lifecycle enforcement, explicit capability requirements, tie policy, and score-level explainability.

The updated policy addresses these limitations by using clear feasibility checks followed by a balanced, tier-aware weighted score. It creates safer and more defensible routing decisions without changing Prompt Profiling.

## Backward Compatibility

Existing routing calls remain supported. The existing prompt, token, legacy-model, and top-N inputs still work with their previous defaults. New enterprise-criticality and required-capability inputs are optional.

The registry retains all existing fields; new fields are additive. Routing responses retain their prior structure and add optional score information. Existing clients can ignore those added fields. Ranking behavior intentionally changes because cost is no longer the dominant decision rule.

## Expected Benefits

- Better T3 routing through stronger quality and reasoning emphasis.
- Better enterprise routing through criticality-aware quality and reliability requirements.
- Safer production behavior through lifecycle, availability, output, and full-context checks.
- Easier future model additions through richer registry metadata and canonical API identifiers.
- More maintainable registry governance through verification and lifecycle status.
- Improved explainability through visible scores, reasons, and rejection details.

## Future Improvements

- Add live provider health, availability, quota, and latency telemetry.
- Refresh pricing automatically from approved provider price cards.
- Add provider-specific deployment and regional-residency validation.
- Calibrate weights and quality floors using evaluation and production outcome data.
- Add controlled shadow routing and A/B evaluation for newly introduced models.
- Add automated registry validation against enabled provider account inventories.
