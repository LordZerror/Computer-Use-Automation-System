from src.artifact.schema import Capability, Checkpoint, InputParam, LocatorStrategy, OutputSpec, Step, TargetSpec


def _sample_capability() -> Capability:
    return Capability(
        id="add_to_cart_checkout",
        name="add_to_cart_checkout",
        goal="Add an item to the cart and reach checkout overview",
        target=TargetSpec(base_url="https://www.saucedemo.com/", app_id="saucedemo"),
        input_params=[
            InputParam(name="password", type="string", sensitive=True),
            InputParam(name="zip_code", type="string"),
        ],
        steps=[
            Step(
                step_id="step_0",
                action="click",
                locators=[LocatorStrategy(kind="test_id", test_id="add-to-cart-sauce-labs-backpack")],
            ),
            Step(step_id="step_1", action="type", param_ref="password"),
        ],
        output_spec=[
            OutputSpec(
                name="total_price",
                type="string",
                source_step_id="step_5",
                extract=LocatorStrategy(kind="test_id", test_id="summary_total_label"),
            )
        ],
        checkpoint=Checkpoint(kind="text_present", text="Checkout: Overview"),
        created_from_run_id="run123",
    )


def test_round_trip_serialization():
    cap = _sample_capability()
    restored = Capability.model_validate_json(cap.model_dump_json())
    assert restored == cap


def test_sensitive_param_never_carries_a_literal_value():
    """A sensitive input param must be supplied at replay time, never baked
    into the artifact as a literal step value."""
    cap = _sample_capability()
    sensitive_names = {p.name for p in cap.input_params if p.sensitive}
    for step in cap.steps:
        if step.param_ref in sensitive_names:
            assert step.value is None
