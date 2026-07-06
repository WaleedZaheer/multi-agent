"""
agents.py
---------
Each function below is a NODE in the LangGraph graph.

The rule every node follows is the same:
    def node(state: AgentState) -> dict:
        ... do some work, maybe call the LLM or a tool ...
        return {only the fields this node is updating}

You do NOT need to return the whole state - LangGraph merges whatever dict
you return into the existing state automatically. This keeps each agent
simple: it only has to think about its own job.
"""

from langchain_core.messages import HumanMessage

from config import get_llm
from state import AgentState
from tools import web_search


# ---------------------------------------------------------------------------
# 1. MANAGER AGENT
# Receives the raw task and breaks it into a brief for the researcher and a
# brief for the writer. This is "task delegation" - the manager doesn't do
# the research or writing itself, it just plans the work.
# ---------------------------------------------------------------------------

MANAGER_PROMPT = """You are the manager of a small content team with a researcher,
a writer, a reviewer, and a publisher working for you.

Task from the client: {task}

Break this down into instructions for your team. Respond in EXACTLY this format:

RESEARCH BRIEF:
- (2-4 bullet points of what the researcher should look up)

WRITING BRIEF:
- (2-3 bullet points on tone, length, audience, and structure for the writer)
"""


def manager_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.2)
    prompt = MANAGER_PROMPT.format(task=state["task"])
    response = llm.invoke([HumanMessage(content=prompt)])

    research_brief, writing_brief = _split_briefs(response.content)

    return {
        "research_brief": research_brief,
        "writing_brief": writing_brief,
        "log": state["log"] + ["Manager: created research and writing briefs"],
    }


def _split_briefs(text: str) -> tuple[str, str]:
    """Small helper to pull the two labelled sections out of the LLM's reply."""
    research_lines, writing_lines = [], []
    current = None

    for line in text.splitlines():
        upper = line.strip().upper()
        if upper.startswith("RESEARCH BRIEF"):
            current = "research"
            continue
        if upper.startswith("WRITING BRIEF"):
            current = "writing"
            continue
        if current == "research":
            research_lines.append(line)
        elif current == "writing":
            writing_lines.append(line)

    research = "\n".join(research_lines).strip()
    writing = "\n".join(writing_lines).strip()

    # fall back to sane defaults if the model didn't follow the format exactly
    return research or text.strip(), writing or "Clear, concise, well-structured, factual tone."


# ---------------------------------------------------------------------------
# 2. RESEARCHER AGENT
# Uses the web_search TOOL to gather raw information, then uses the LLM to
# synthesize that raw information into something the writer can actually use.
# This is the classic "tool call + LLM synthesis" pattern.
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.2)
    raw_results = web_search(state["task"])

    prompt = f"""You are a research analyst. Using the brief and the raw search
results below, write a clear, well-organized summary of the key facts, figures,
and viewpoints a writer would need. Only use information present in the search
results - do not invent facts.

Research brief:
{state['research_brief']}

Raw search results:
{raw_results}

Write the synthesized findings now:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "research_findings": response.content,
        "log": state["log"] + ["Researcher: gathered and synthesized web results"],
    }


# ---------------------------------------------------------------------------
# 3. WRITER AGENT
# Writes the actual content using the research findings and the writing
# brief. If this is a revision (review_feedback is not empty), it also folds
# in the reviewer's notes - this is what makes the loop actually improve the
# draft instead of just repeating it.
# ---------------------------------------------------------------------------

def writer_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.6)  # a bit more creative than the other agents

    revision_note = ""
    if state.get("review_feedback"):
        revision_note = f"""

This is a REVISION. The previous draft received this feedback - address it directly:
{state['review_feedback']}"""

    prompt = f"""You are a content writer. Write content for the task below, using
the research findings and writing brief provided.

Task: {state['task']}

Writing brief:
{state['writing_brief']}

Research findings:
{state['research_findings']}{revision_note}

Write the full piece now. No preamble, no meta-commentary - just the content:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft": response.content,
        "log": state["log"] + [f"Writer: produced draft (attempt {state['revision_count'] + 1})"],
    }


# ---------------------------------------------------------------------------
# 4. REVIEWER AGENT
# Critiques the draft and makes a decision: APPROVE or REVISE. This decision
# is what the conditional edge in graph.py reads to decide where to route
# next. Notice the reviewer doesn't call any Python function to "loop" -
# it just writes its verdict into the state, and the GRAPH decides what
# happens next based on that.
# ---------------------------------------------------------------------------

def reviewer_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.1)  # low temperature - we want a consistent judge

    prompt = f"""You are an exacting editor. Review the draft below against the
task and writing brief. Check for factual grounding, clarity, structure, and tone.

Task: {state['task']}
Writing brief: {state['writing_brief']}

Draft:
{state['draft']}

Respond in EXACTLY this format:
DECISION: APPROVE
or
DECISION: REVISE

FEEDBACK: (specific, actionable notes the writer should act on. If approving,
briefly say why it's good.)"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content
    text_upper = text.upper()

    approved = "DECISION: APPROVE" in text_upper

    if "FEEDBACK:" in text_upper:
        split_point = text_upper.index("FEEDBACK:")
        feedback = text[split_point + len("FEEDBACK:"):].strip()
    else:
        feedback = text.strip()

    return {
        "approved": approved,
        "review_feedback": feedback,
        "revision_count": state["revision_count"] + 1,
        "log": state["log"] + [f"Reviewer: {'approved' if approved else 'requested a revision'}"],
    }


# ---------------------------------------------------------------------------
# 5. PUBLISHER AGENT
# No LLM call needed here - it just formats the approved draft into a clean
# final document. Not every node has to use AI; some just do plain
# programming work. That's a normal and good design choice.
# ---------------------------------------------------------------------------

def publisher_node(state: AgentState) -> dict:
    header = f"# {state['task']}\n\n"
    meta = f"*Produced by a 5-agent pipeline - {state['revision_count']} revision round(s).*\n\n---\n\n"
    final = header + meta + state["draft"].strip() + "\n"

    return {
        "final_output": final,
        "log": state["log"] + ["Publisher: formatted the final output"],
    }
