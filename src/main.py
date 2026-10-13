
from pathlib import Path

from llm_client import generate_response


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENT_FILE = PROJECT_ROOT / "data" / "requirement.txt"


def main() -> None:
    if not REQUIREMENT_FILE.exists():
        raise FileNotFoundError(
            f"Requirement file not found: {REQUIREMENT_FILE}"
        )

    requirement = REQUIREMENT_FILE.read_text(encoding="utf-8")

    prompt = f"""
You are an experienced Software QA Engineer.

Analyze the following software requirement.

Provide:
1. A brief requirement summary.
2. Positive test scenarios.
3. Negative test scenarios.
4. Boundary and edge cases.
5. Security-related test scenarios.
6. Missing details or ambiguities in the requirement.

Do not assume unspecified behavior is confirmed.
Clearly identify assumptions and missing information.

Software requirement:
{requirement}
"""

    print("Analyzing the software requirement with Gemini...\n")

    result = generate_response(prompt)

    print("=" * 60)
    print("AI QA COPILOT — REQUIREMENT ANALYSIS")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
