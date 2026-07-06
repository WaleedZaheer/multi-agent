"""
main.py
-------
Run this file to actually use the team:

    python main.py "Write a short blog post about the benefits of retinoscopy"

Or run it with no arguments and it will ask you for a task.
"""

import sys
from datetime import datetime

from graph import build_graph


def run(task: str):
    app = build_graph()

    # This is the "clipboard" at the very start - only the task is filled in,
    # everything else starts empty/default and gets filled in as agents run.
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

    print(f"\nTask received: {task}")
    print("Running the team (manager -> researcher -> writer -> reviewer -> publisher)...\n")

    result = app.invoke(initial_state)

    print("=" * 60)
    print("PIPELINE LOG")
    print("=" * 60)
    for entry in result["log"]:
        print(f"- {entry}")

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result["final_output"])

    filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(result["final_output"])
    print(f"\nSaved to {filename}")


if __name__ == "__main__":
    task_arg = " ".join(sys.argv[1:])
    task = task_arg or input("Enter a task for the team: ")
    run(task)
