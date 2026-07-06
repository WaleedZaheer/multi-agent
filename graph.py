"""
graph.py
--------
This is where the "team" actually gets assembled. Three steps, every time:
  1. Create a StateGraph, telling it what shape the state has.
  2. Register each agent as a node.
  3. Wire nodes together with edges (and one CONDITIONAL edge for the loop).
"""

from langgraph.graph import StateGraph, START, END

from state import AgentState
from agents import (
    manager_node,
    researcher_node,
    writer_node,
    reviewer_node,
    publisher_node,
)

# Safety valve: without this, a stubborn reviewer and a struggling writer
# could loop forever. After this many attempts, we force-publish whatever
# we have. Kept as a default so main.py works with no arguments; app.py
# passes its own value from a slider.
DEFAULT_MAX_REVISIONS = 2


def build_graph(max_revisions: int = DEFAULT_MAX_REVISIONS):
    def review_router(state: AgentState) -> str:
        """
        The "brain" of the conditional edge. LangGraph calls this after the
        reviewer node runs, and whatever string it returns tells the graph
        which node to go to next. Defined INSIDE build_graph so it can "see"
        max_revisions via closure.
        """
        if state["approved"]:
            return "publisher"
        if state["revision_count"] >= max_revisions:
            return "publisher"  # give up gracefully instead of looping forever
        return "writer"

    graph = StateGraph(AgentState)

    # Step 1: register every agent as a node
    graph.add_node("manager", manager_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("publisher", publisher_node)

    # Step 2: the straightforward, one-way wiring
    graph.add_edge(START, "manager")
    graph.add_edge("manager", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")

    # Step 3: the one INTERESTING piece of wiring - the review loop.
    # After "reviewer" runs, call review_router(state). Whatever it returns
    # ("writer" or "publisher") is where we go next.
    graph.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "writer": "writer",
            "publisher": "publisher",
        },
    )

    graph.add_edge("publisher", END)

    # .compile() turns the wiring diagram into a runnable object
    return graph.compile()
