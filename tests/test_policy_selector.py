from core.policy_selector import decide_leading_brain


def test_policy_selector_prefers_right_keywords():
    ctx = {"novelty": 0.3, "memory": "dream symbol"}
    assert decide_leading_brain("Imagine a mythic dream", ctx) == "right"


def test_policy_selector_balances_recent_lead():
    ctx = {"novelty": 0.4, "recent_lead": "left"}
    assert decide_leading_brain("Tell me something", ctx) == "right"


def test_policy_selector_defaults_left_for_analysis():
    ctx = {"novelty": 0.9}
    assert decide_leading_brain("Please analyze this dataset", ctx) == "left"
