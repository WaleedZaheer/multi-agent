"""
app.py
------
Streamlit UI for the multi-agent team. Run with:

    streamlit run app.py

This file does NOT contain any agent logic - it only calls build_graph()
from graph.py and displays what comes back. That separation matters: your
agent logic stays testable and reusable (main.py, test_graph_logic.py, and
this UI all share the exact same graph.py / agents.py code).
"""

import streamlit as st

from graph import build_graph, DEFAULT_MAX_REVISIONS

st.set_page_config(page_title="Multi-agent content team", page_icon="🤝", layout="centered")

STEP_LABELS = {
    "manager": "Manager - planning the work",
    "researcher": "Researcher - gathering information",
    "writer": "Writer - drafting content",
    "reviewer": "Reviewer - checking the draft",
    "publisher": "Publisher - formatting the final output",
}


def render_step(node_name: str, update: dict, writer_attempts: list):
    """Renders one completed agent step as a collapsible status block."""
    label = STEP_LABELS.get(node_name, node_name)

    if node_name == "writer":
        writer_attempts[0] += 1
        label += f" (attempt {writer_attempts[0]})"

    with st.status(label, state="complete"):
        if node_name == "manager":
            st.markdown("**Research brief**")
            st.text(update["research_brief"])
            st.markdown("**Writing brief**")
            st.text(update["writing_brief"])

        elif node_name == "researcher":
            st.markdown("**Research findings**")
            st.text(update["research_findings"])

        elif node_name == "writer":
            st.markdown("**Draft**")
            st.text(update["draft"])

        elif node_name == "reviewer":
            verdict = "Approved" if update["approved"] else "Sent back for revision"
            st.markdown(f"**Verdict:** {verdict}")
            st.markdown("**Feedback**")
            st.text(update["review_feedback"])

        elif node_name == "publisher":
            st.markdown("Final output formatted below.")


def main():
    st.title("Multi-agent content team")
    st.caption("Manager -> Researcher -> Writer -> Reviewer -> Publisher, built with LangGraph + Groq")

    with st.sidebar:
        st.subheader("Settings")
        max_revisions = st.slider(
            "Max revision rounds",
            min_value=0,
            max_value=4,
            value=DEFAULT_MAX_REVISIONS,
            help="How many times the Reviewer can send the draft back to the Writer before it's force-published.",
        )
        st.divider()
        st.caption(
            "GROQ_API_KEY is read from your .env file, or from Streamlit "
            "secrets if this app is deployed."
        )

    task = st.text_area(
        "What should the team work on?",
        placeholder="e.g. Write a short, friendly explainer on how retinoscopy works",
        height=90,
    )

    run_clicked = st.button("Run the team", type="primary", disabled=not task.strip())

    if run_clicked:
        try:
            app = build_graph(max_revisions=max_revisions)
        except Exception as e:
            st.error(f"Couldn't build the graph: {e}")
            return

        initial_state = {
            "task": task,
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

        st.divider()
        st.subheader("Team activity")

        progress_area = st.container()
        current_state = dict(initial_state)
        writer_attempts = [0]  # mutable counter, shared with render_step

        try:
            with st.spinner("The team is working on your task..."):
                for chunk in app.stream(initial_state):
                    node_name, update = next(iter(chunk.items()))
                    current_state.update(update)
                    with progress_area:
                        render_step(node_name, update, writer_attempts)
        except Exception as e:
            st.error(
                f"Something went wrong while running the team: {e}\n\n"
                "Check that GROQ_API_KEY is set correctly in your .env file."
            )
            return

        st.divider()
        st.subheader("Final output")
        st.markdown(current_state["final_output"])

        st.download_button(
            "Download as Markdown",
            data=current_state["final_output"],
            file_name="team_output.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
