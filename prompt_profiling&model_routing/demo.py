import os
import sys
import argparse
from pathlib import Path

# Add current directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from router import route_model

def format_result(result) -> str:
    p = result.prompt_profile
    lines = []
    lines.append("=" * 60)
    lines.append("  PROMPT PROFILE")
    lines.append("=" * 60)
    lines.append(f"  Derived Tier    : {p.derived_tier} (Composite Score: {p.complexity_score:.4f})")
    lines.append(f"  Resolved Tier   : {result.resolved_tier} " + (f"({result.escalation_reason})" if result.tier_escalated else "(no escalation)"))
    lines.append(f"  Confidence      : {p.confidence:.4f}")
    lines.append(f"  Domain          : {p.domain}")
    lines.append(f"  Intent / Task   : {p.intent} / {p.task_type}")
    lines.append(f"  Dimensions      : D1={p.d1:.2f} | D2={p.d2:.2f} | D3={p.d3:.2f} | D4={p.d4:.2f} | D5={p.d5:.2f}")
    lines.append(f"  Reasoning Chain : {'YES' if p.reasoning_chain_detected else 'NO'}")
    lines.append(f"  Research Signals: {', '.join(p.research_signals) if p.research_signals else 'None'}")
    lines.append(f"  Token Estimates : Input={p.input_token_count:,} | Est. Output={p.est_output_tokens:,}")
    lines.append("")

    lines.append("=" * 60)
    lines.append(f"  TOP {len(result.recommendations)} RECOMMENDED MODELS (by cost)")
    lines.append("=" * 60)

    if not result.recommendations:
        lines.append("  [NO MODELS SURVIVED FILTERS]")
    else:
        for r in result.recommendations:
            lines.append(f"  #{r.rank} {r.model_id:<25} Cost: ${r.estimated_cost_usd:.6f}  [{r.provider} | Tier {r.tier}]")
            for reason in r.reasons:
                lines.append(f"      - {reason}")
            lines.append("")

    if result.warnings:
        lines.append("=" * 60)
        lines.append("  WARNINGS")
        lines.append("=" * 60)
        for w in result.warnings:
            lines.append(f"  ! {w}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"  REJECTIONS SUMMARY ({len(result.rejections)} models excluded)")
    lines.append("=" * 60)
    for model_id, reason in result.rejections.items():
        lines.append(f"  x {model_id:<25} : {reason}")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Prompt Profiling & Model Routing CLI Demo")
    parser.add_argument("prompt", type=str, nargs="?", help="Input prompt text to profile and route")
    parser.add_argument("--max-tokens", type=int, default=None, help="Caller-supplied max output tokens override")
    parser.add_argument("--include-legacy", action="store_true", help="Include legacy models in candidates")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top model recommendations to display")
    parser.add_argument("--pkl", type=str, default="prompt_profiler.pkl", help="Path to prompt profiler pickle bundle")
    parser.add_argument("--registry", type=str, default="model_registry_v3.json", help="Path to model registry v3 JSON")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.resolve()
    pkl_path = str(base_dir / args.pkl) if not os.path.isabs(args.pkl) else args.pkl
    registry_path = str(base_dir / args.registry) if not os.path.isabs(args.registry) else args.registry

    if not args.prompt:
        # Default sample prompt if none provided
        sample_prompt = (
            "Design a multi-cloud GenAI governance architecture for a Fortune 500 company, "
            "including compliance risks, cost management, and vendor evaluation criteria."
        )
        print(f"\n[No prompt provided. Running default sample prompt]:\n\"{sample_prompt}\"\n")
        prompt_text = sample_prompt
    else:
        prompt_text = args.prompt

    result = route_model(
        prompt=prompt_text,
        max_tokens=args.max_tokens,
        pkl_path=pkl_path,
        registry_path=registry_path,
        include_legacy=args.include_legacy,
        top_n=args.top_n,
    )

    print(format_result(result))


if __name__ == "__main__":
    main()
