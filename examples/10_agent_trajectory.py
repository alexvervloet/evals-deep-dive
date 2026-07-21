"""
Example 10: evaluating an agent's TRAJECTORY, not just its answer. (offline)

Everything so far graded a single output string. But an *agent* takes a series of
steps. It calls tools, reads results, and decides what to do next, and a right final
answer can hide a broken process: the lucky guess, the forbidden tool call, the
agent that took 9 steps to do a 2-step job. So you evaluate the **trajectory**.

The shift: the task's "output" is now a structured **trace** (the steps it took +
its answer), and you write scorers that grade the *process* on several axes:

  - final_answer   did it end up correct? (necessary, not sufficient)
  - used_tool      did it call the tool the task actually requires?
  - no_forbidden   did it avoid tools it shouldn't touch (e.g. delete)?
  - efficient      did it finish within a sane step budget?

This is the same dataset -> task -> scorer -> report loop from example 01; the
only new idea is that outputs (and therefore scorers) are richer. Fully offline:
the three "agents" are canned so you can see the scores without a model.

Run it:

    python examples/10_agent_trajectory.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals
from evals import Example, Score

# --- The tasks. metadata carries the trajectory requirements we'll grade on. ---
DATASET = [
    Example(
        input="What's the weather in Paris, in Fahrenheit?",
        expected="59",
        metadata={"must_use": "get_weather", "forbidden": "delete_file", "budget": 3},
    ),
    Example(
        input="How many users signed up last week?",
        expected="1240",
        metadata={"must_use": "query_db", "forbidden": "delete_file", "budget": 2},
    ),
]


# --- Three "agents under test." Each returns a TRACE: the steps it took (tool
# calls) plus its final answer, serialized as JSON (the task's "output"). ---
def good_agent(task: str) -> str:
    if "weather" in task:
        steps = [{"tool": "get_weather", "args": "Paris"}, {"tool": "convert", "args": "C->F"}]
        return json.dumps({"steps": steps, "answer": "59"})
    steps = [{"tool": "query_db", "args": "signups last week"}]
    return json.dumps({"steps": steps, "answer": "1240"})


def lucky_agent(task: str) -> str:
    # Right answer, but it never actually used the required tool. It guessed.
    answer = "59" if "weather" in task else "1240"
    return json.dumps({"steps": [{"tool": "think", "args": "..."}], "answer": answer})


def reckless_agent(task: str) -> str:
    # Correct answer, but wasteful AND it called a forbidden, destructive tool.
    if "weather" in task:
        steps = [{"tool": "get_weather", "args": "Paris"}, {"tool": "delete_file", "args": "tmp"},
                 {"tool": "get_weather", "args": "Paris"}, {"tool": "convert", "args": "C->F"}]
        return json.dumps({"steps": steps, "answer": "59"})
    steps = [{"tool": "query_db", "args": "x"}, {"tool": "delete_file", "args": "y"},
             {"tool": "query_db", "args": "signups"}]
    return json.dumps({"steps": steps, "answer": "1240"})


# --- Trajectory scorers: each grades one axis of the trace. ---
def _trace(output: str) -> dict:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"steps": [], "answer": ""}


def final_answer(output: str, ex: Example) -> Score:
    ok = _trace(output)["answer"].strip() == (ex.expected or "")
    return Score(passed=ok, score=1.0 if ok else 0.0, detail="answer matches" if ok else "wrong answer")


def used_tool(output: str, ex: Example) -> Score:
    tools = [s["tool"] for s in _trace(output)["steps"]]
    ok = ex.metadata["must_use"] in tools
    return Score(passed=ok, score=1.0 if ok else 0.0,
                 detail=f"used {ex.metadata['must_use']}" if ok else f"never called {ex.metadata['must_use']}")


def no_forbidden(output: str, ex: Example) -> Score:
    tools = [s["tool"] for s in _trace(output)["steps"]]
    bad = ex.metadata["forbidden"] in tools
    return Score(passed=not bad, score=0.0 if bad else 1.0,
                 detail=f"called forbidden {ex.metadata['forbidden']}!" if bad else "no forbidden tools")


def efficient(output: str, ex: Example) -> Score:
    n = len(_trace(output)["steps"])
    ok = n <= ex.metadata["budget"]
    return Score(passed=ok, score=1.0 if ok else 0.0, detail=f"{n} steps (budget {ex.metadata['budget']})")


SCORERS = {
    "final_answer": final_answer,
    "used_tool": used_tool,
    "no_forbidden": no_forbidden,
    "efficient": efficient,
}

if __name__ == "__main__":
    print("Evaluating three agents on the SAME tasks; all reach the right answer.\n")
    for name, agent in [("good", good_agent), ("lucky (guessed)", lucky_agent),
                        ("reckless (wasteful + forbidden tool)", reckless_agent)]:
        report = evals.run_eval(task=agent, dataset=DATASET, scorers=SCORERS)
        print(f"=== {name} agent ===")
        report.print_summary()
        print()

    print(
        "Takeaway: final-answer accuracy hides process failures. The 'lucky' agent\n"
        "scores 100% on the answer but 0% on tool use: it guessed. The 'reckless' one\n"
        "is correct but called a destructive tool and blew the step budget. Grading the\n"
        "TRAJECTORY catches the agents that are right for the wrong reasons."
    )
