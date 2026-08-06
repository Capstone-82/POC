"""
Token estimator for Lambda context window checks.
Uses word-count approximation (words * 1.35) when tiktoken is unavailable.
"""

def estimate_tokens(text: str) -> int:
    """Estimate token count from text using word count approximation."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.35))


def check_context_limits(
    prompt: str,
    max_output_tokens: int,
    limits: dict,
) -> dict:
    """
    Check if a prompt fits within configured context window limits.

    Args:
        prompt: The input prompt text.
        max_output_tokens: Caller-specified max output tokens (or 0 for default).
        limits: Dict with keys max_input_tokens, max_output_tokens, max_total_tokens.

    Returns:
        Dict with keys:
            allowed (bool)
            estimated_input_tokens (int)
            estimated_output_tokens (int)
            estimated_total_tokens (int)
            violation (str | None) — which limit was breached, if any
            message (str | None)
    """
    max_in  = limits.get("max_input_tokens", 8000)
    max_out = limits.get("max_output_tokens", 4096)
    max_tot = limits.get("max_total_tokens", 10000)

    est_input  = estimate_tokens(prompt)
    est_output = max_output_tokens if max_output_tokens > 0 else max_out
    est_total  = est_input + est_output

    if est_input > max_in:
        return {
            "allowed": False,
            "estimated_input_tokens": est_input,
            "estimated_output_tokens": est_output,
            "estimated_total_tokens": est_total,
            "violation": "max_input_tokens",
            "message": (
                f"Estimated {est_input:,} input tokens exceeds limit of "
                f"{max_in:,} for this application."
            ),
        }

    if est_output > max_out:
        return {
            "allowed": False,
            "estimated_input_tokens": est_input,
            "estimated_output_tokens": est_output,
            "estimated_total_tokens": est_total,
            "violation": "max_output_tokens",
            "message": (
                f"Requested {est_output:,} output tokens exceeds limit of "
                f"{max_out:,} for this application."
            ),
        }

    if est_total > max_tot:
        return {
            "allowed": False,
            "estimated_input_tokens": est_input,
            "estimated_output_tokens": est_output,
            "estimated_total_tokens": est_total,
            "violation": "max_total_tokens",
            "message": (
                f"Estimated total {est_total:,} tokens exceeds limit of "
                f"{max_tot:,} for this application."
            ),
        }

    return {
        "allowed": True,
        "estimated_input_tokens": est_input,
        "estimated_output_tokens": est_output,
        "estimated_total_tokens": est_total,
        "violation": None,
        "message": None,
    }
