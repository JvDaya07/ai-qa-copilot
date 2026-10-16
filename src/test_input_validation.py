
from test_case_generator import validate_requirement


def run_validation_tests() -> None:
    # 1. Empty requirement
    try:
        validate_requirement("   ")
        raise AssertionError("Empty input should be rejected")
    except ValueError as exc:
        print(f"PASS — Empty input: {exc}")

    # 2. Too-short requirement
    try:
        validate_requirement("Login")
        raise AssertionError("Short input should be rejected")
    except ValueError as exc:
        print(f"PASS — Short input: {exc}")

    # 3. Normal requirement
    normal = "Users can log in using their email and password."
    validate_requirement(normal)
    print("PASS — Normal input accepted")

    # 4. Long but permitted requirement
    long_text = (
        "Users can log in using their email and password. "
        * 100
    )
    validate_requirement(long_text)
    print(
        f"PASS — Long input accepted ({len(long_text)} characters)"
    )

    # 5. Oversized requirement
    oversized = "A" * 10_001

    try:
        validate_requirement(oversized)
        raise AssertionError("Oversized input should be rejected")
    except ValueError as exc:
        print(f"PASS — Oversized input: {exc}")


if __name__ == "__main__":
    run_validation_tests()
