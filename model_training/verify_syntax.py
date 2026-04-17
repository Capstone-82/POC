from __future__ import annotations

import ast
import re
from pathlib import Path

from dotenv import load_dotenv

from generate_avg_accuracy_scores import get_supabase_client


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def check_code_syntax(response: str) -> bool | None:
    """
    True  -> valid Python syntax in at least one code block
    False -> Python code block exists but fails parsing
    None  -> no code block found
    """
    blocks = CODE_BLOCK_RE.findall(response or "")
    if not blocks:
        return None

    for block in blocks:
        snippet = block.strip()
        if not snippet:
            continue
        try:
            ast.parse(snippet)
            return True
        except SyntaxError:
            return False
    return None


def fetch_pending_rows(supabase) -> list[dict]:
    page_size = 1000
    start = 0
    rows: list[dict] = []
    while True:
        batch = (
            supabase.table("benchmark_results")
            .select("id,response")
            .eq("use_case", "code-generation")
            .or_("syntax_checked.is.false,syntax_checked.is.null")
            .range(start, start + page_size - 1)
            .execute()
            .data
            or []
        )
        if not batch:
            break
        rows.extend(batch)
        start += page_size
    return rows


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    supabase = get_supabase_client()
    rows = fetch_pending_rows(supabase)
    print(f"Checking syntax for {len(rows)} code-generation rows.")

    updated = 0
    for index, row in enumerate(rows, start=1):
        syntax_pass = check_code_syntax(str(row.get("response", "")))
        supabase.table("benchmark_results").update(
            {
                "syntax_pass": syntax_pass,
                "syntax_checked": True,
            }
        ).eq("id", row["id"]).execute()
        updated += 1
        if index % 100 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)}")

    print(f"Updated {updated} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
