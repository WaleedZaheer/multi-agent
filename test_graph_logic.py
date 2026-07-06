"""
Not part of the tutorial project - this is just how I (Claude) verified the
graph wiring and revision loop actually work before handing you the code.
Mocks get_llm() and web_search() so no API key / internet is needed.
"""

import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/home/claude/multi_agent_team")

call_count = {"reviewer": 0}


class FakeResponse:
    def __init__(self, content):
        self.content = content


def fake_llm_invoke(messages):
    prompt = messages[0].content

    if "manager of a small content team" in prompt:
        return FakeResponse(
            "RESEARCH BRIEF:\n- look up X\n- look up Y\n\nWRITING BRIEF:\n- keep it short\n- friendly tone"
        )
    if "research analyst" in prompt:
        return FakeResponse("Fake synthesized research findings about the topic.")
    if "content writer" in prompt:
        if "REVISION" in prompt:
            return FakeResponse("Improved draft addressing the feedback.")
        return FakeResponse("Initial draft of the content.")
    if "exacting editor" in prompt:
        call_count["reviewer"] += 1
        if call_count["reviewer"] == 1:
            return FakeResponse("DECISION: REVISE\n\nFEEDBACK: Needs more detail in paragraph 2.")
        return FakeResponse("DECISION: APPROVE\n\nFEEDBACK: Looks good now.")
    return FakeResponse("fallback")


fake_llm = MagicMock()
fake_llm.invoke.side_effect = fake_llm_invoke

with patch("config.get_llm", return_value=fake_llm), \
     patch("agents.get_llm", return_value=fake_llm), \
     patch("agents.web_search", return_value="fake search result 1\nfake search result 2"):

    from graph import build_graph

    app = build_graph()
    initial_state = {
        "task": "Write a short post about retinoscopy basics",
        "research_brief": "",
        "writing_brief": "",
        "research_findings": "",
        "draft": "",
        "review_feedback": "",
        "approved": False,
        "revision_count": 0,
        "final_output": "",
        "log": [],
    }

    result = app.invoke(initial_state)

    print("LOG:")
    for entry in result["log"]:
        print(" -", entry)

    assert result["approved"] is True, "should end approved"
    assert result["revision_count"] == 2, f"expected 2 review rounds, got {result['revision_count']}"
    assert "Improved draft" in result["draft"], "final draft should be the revised one"
    assert result["final_output"].startswith("# Write a short post"), "publisher should format with title"

    print("\nALL ASSERTIONS PASSED - graph wiring and revision loop work correctly.")
