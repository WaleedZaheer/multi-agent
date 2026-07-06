"""
state.py
--------
This is the single most important concept in LangGraph: the STATE.

Think of it as a clipboard that gets passed from agent to agent. Every agent
reads whatever fields it needs off the clipboard, does its job, and writes
its results back onto the clipboard before passing it to the next agent.

Because it's a TypedDict, your editor and LangGraph both know exactly what
fields exist and what type each one is - this catches typos early
(e.g. if you write state["darft"] instead of state["draft"], you'll notice).
"""

from typing import TypedDict, List


class AgentState(TypedDict):
    # --- input ---
    task: str                  # the original request from the user

    # --- manager's output ---
    research_brief: str        # what the researcher should look up
    writing_brief: str         # tone/length/structure instructions for the writer

    # --- researcher's output ---
    research_findings: str     # synthesized facts the writer will use

    # --- writer's output ---
    draft: str                 # the current draft (gets overwritten on each revision)

    # --- reviewer's output ---
    review_feedback: str       # specific notes for the writer to act on
    approved: bool             # True once the reviewer signs off
    revision_count: int        # how many times we've been through the writer/reviewer loop

    # --- publisher's output ---
    final_output: str          # the fully formatted, ready-to-share result

    # --- shared across everyone ---
    log: List[str]             # a running trail of "who did what", for teaching/debugging
