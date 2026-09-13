import pytest

from src.safety.policy import PolicyViolation, load_policy


def test_allowed_domain_passes():
    policy = load_policy()
    policy.check_domain("https://www.saucedemo.com/inventory.html")  # must not raise


def test_disallowed_domain_blocked():
    policy = load_policy()
    with pytest.raises(PolicyViolation):
        policy.check_domain("https://evil.example.com/")


def test_disallowed_action_type_blocked():
    policy = load_policy()
    with pytest.raises(PolicyViolation):
        policy.check_action_type("download_file")


def test_risky_control_classified_by_text_marker():
    policy = load_policy()
    assert policy.classify_risk("Finish") == "risky"
    assert policy.classify_risk("Continue") == "safe"


def test_redact_params_masks_sensitive_values():
    policy = load_policy()
    out = policy.redact_params(
        {"username": "standard_user", "password": "secret_sauce"},
        {"username": False, "password": True},
    )
    assert out["username"] == "standard_user"
    assert out["password"] == "***REDACTED***"


def test_redact_params_masks_by_name_even_if_not_declared():
    policy = load_policy()
    out = policy.redact_params({"card_number": "4111111111111111"}, {})
    assert out["card_number"] == "***REDACTED***"
