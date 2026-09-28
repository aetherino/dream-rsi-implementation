"""Versioned objectives. New searches minimize total extra prompt computation."""
TOTAL_EXTRA = "total_extra_v2"
LEGACY_MACRO = "mean_relative_v1"
SCORING_MODES = (TOTAL_EXTRA, LEGACY_MACRO)


def score_recomputation(candidate_extra, lru_extra, scoring=TOTAL_EXTRA):
    if scoring not in SCORING_MODES:
        raise ValueError(f"Unknown scoring mode: {scoring}")
    if not candidate_extra or len(candidate_extra) != len(lru_extra):
        raise ValueError("Scoring requires matching nonempty scenario counts")
    if any(type(n) is not int or n < 0 for n in [*candidate_extra, *lru_extra]):
        raise ValueError("Extra-token counts must be nonnegative integers")
    saved = sum(lru_extra) - sum(candidate_extra)
    denominator = max(1, sum(lru_extra))
    macro = sum((reference - actual) / max(1, reference)
                for actual, reference in zip(candidate_extra, lru_extra)) / len(candidate_extra)
    return {"scoring": scoring, "score": saved / denominator if scoring == TOTAL_EXTRA else macro,
            "total_extra_computed_tokens": sum(candidate_extra),
            "total_lru_extra_computed_tokens": sum(lru_extra), "saved_prompt_tokens": saved,
            "score_denominator_tokens": denominator if scoring == TOTAL_EXTRA else None,
            "legacy_mean_relative_improvement": macro,
            "score_contributions": [(reference - actual) / denominator if scoring == TOTAL_EXTRA
                                    else (reference - actual) / max(1, reference) / len(candidate_extra)
                                    for actual, reference in zip(candidate_extra, lru_extra)]}


def objective_description(scoring):
    if scoring == TOTAL_EXTRA:
        return ("The task score is (SUM of LRU extra tokens - SUM of candidate extra tokens) / "
                "max(1, SUM of LRU extra tokens), summed across all scenarios BEFORE dividing. "
                "Every saved token has equal value. A positive score requires fewer TOTAL recomputed tokens. "
                "Per-scenario relative improvements are diagnostics, not the optimization objective. "
                "A scenario where LRU has zero extra tokens uses the same suite-wide denominator as every other scenario.")
    if scoring == LEGACY_MACRO:
        return ("The task score is the equally weighted mean of (LRU extra tokens - candidate extra tokens) / "
                "max(1, LRU extra tokens) across scenarios. This legacy score does not minimize total tokens.")
    raise ValueError(f"Unknown scoring mode: {scoring}")
