import pytest

from src.safety.policy import PolicyViolation, load_policy


def test_allowed_domain_passes():
    policy = load_policy()
    policy.check_domain("https://www.saucedemo.com/inventory.html")  # must not raise


def test_disallowed_domain_blocked():
    policy = load_policy()
    with pytest.raises(PolicyViolation):
        policy.check_domain("https://evil.example.com/")


def test_file_url_always_allowed():
    """Local fixtures (the vision-fallback demo target) aren't the
    exfiltration risk a remote-domain allowlist exists to prevent."""
    policy = load_policy()
    policy.check_domain("file:///tmp/fixtures/canvas_button.html")  # must not raise


def test_disallowed_action_type_blocked():
    policy = load_policy()
    with pytest.raises(PolicyViolation):
        policy.check_action_type("download_file")


def test_risky_control_classified_by_text_marker():
    policy = load_policy()
    assert policy.classify_risk("Finish") == "risky"
    assert policy.classify_risk("Continue") == "safe"


def test_classify_action_risk_hover_uses_the_same_text_marker_rule():
    policy = load_policy()
    assert policy.classify_action_risk("hover", "Delete Options") == "risky"
    assert policy.classify_action_risk("hover", "Account") == "safe"


def test_classify_action_risk_folds_value_for_select_option():
    """A neutrally-named dropdown with a risky option must still be caught
    -- the control's own name alone would miss it."""
    policy = load_policy()
    assert policy.classify_action_risk("select_option", "Account Actions", "Delete") == "risky"
    assert policy.classify_action_risk("select_option", "Account Actions", "View Profile") == "safe"


def test_classify_action_risk_folds_value_for_keypress():
    """Regression: keypress used to classify only the focused element's own
    name, so e.g. pressing the Delete key on a neutrally-named element was
    never blocked even though the identical word on a button would be."""
    policy = load_policy()
    assert policy.classify_action_risk("keypress", "row-42", "Delete") == "risky"
    assert policy.classify_action_risk("keypress", "row-42", "Enter") == "safe"


def test_classify_action_risk_click_uses_control_name_only():
    policy = load_policy()
    assert policy.classify_action_risk("click", "Finish") == "risky"
    assert policy.classify_action_risk("click", "Continue") == "safe"


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
