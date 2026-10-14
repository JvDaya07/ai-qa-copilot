
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import json
from pathlib import Path

from llm_client import generate_structured_response


class TestCase(BaseModel):
    """Schema for one AI-generated QA test case."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=5)
    type: Literal[
        "positive",
        "negative",
        "boundary",
        "security",
        "regression",
    ]
    steps: list[str] = Field(min_length=1)
    expected_result: str = Field(min_length=5)
    priority: Literal["high", "medium", "low"]


class TestCaseBatch(BaseModel):
    """Schema for a collection of test cases."""

    model_config = ConfigDict(extra="forbid")

    test_cases: list[TestCase] = Field(min_length=1)
    
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENT_FILE = PROJECT_ROOT / "data" / "requirement.txt"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "test_cases.json"


GEMINI_TEST_CASE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "test_cases": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "type": {
                        "type": "STRING",
                        "enum": [
                            "positive",
                            "negative",
                            "boundary",
                            "security",
                            "regression",
                        ],
                    },
                    "steps": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "expected_result": {"type": "STRING"},
                    "priority": {
                        "type": "STRING",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "title",
                    "type",
                    "steps",
                    "expected_result",
                    "priority",
                ],
            },
        }
    },
    "required": ["test_cases"],
}



def generate_test_cases(requirement: str) -> TestCaseBatch:
    """Generate and validate test cases for a requirement."""

    prompt = f"""
You are a senior software QA engineer.

Generate 10 distinct test cases for the software requirement below.

Include a balanced mix of:
- positive
- negative
- boundary
- security
- regression

Rules:
1. Every test case must have all required fields.
2. Every test case must contain actionable steps.
3. Expected results must be observable and testable.
4. Do not invent unspecified business rules.
5. If behavior is unclear, identify the assumption in the
   expected result or test steps.
6. Choose priority based on business and security risk.
7. Avoid duplicate test cases.

Requirement:
{requirement}
"""

    result = generate_structured_response(
    prompt,
    GEMINI_TEST_CASE_SCHEMA,
)

    # Validate locally even when the provider supplies a schema.
    return TestCaseBatch.model_validate(result)


def main() -> None:
    if not REQUIREMENT_FILE.exists():
        raise FileNotFoundError(
            f"Requirement file not found: {REQUIREMENT_FILE}"
        )

    requirement = REQUIREMENT_FILE.read_text(encoding="utf-8")

    print("Generating structured test cases...\n")

    batch = generate_test_cases(requirement)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        batch.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"Generated {len(batch.test_cases)} test cases.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nPreview:")
    print(
        json.dumps(
            batch.test_cases[0].model_dump(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
