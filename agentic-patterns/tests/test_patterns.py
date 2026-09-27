import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_patterns.prompt_chaining import Step, run_chain
from agentic_patterns.routing import Route, classify, route
from agentic_patterns.parallelization import section, vote
from agentic_patterns.orchestrator_workers import orchestrate
from agentic_patterns.evaluator_optimizer import refine


# ---------- prompt_chaining ----------

def test_run_chain_passes_output_forward_through_steps():
    def stub(system, prompt, temperature):
        return f"[{system[:6]}] processed: {prompt}"

    steps = [
        Step(name="outline", system_prompt="outline", build_prompt=lambda x: x),
        Step(name="draft", system_prompt="draft", build_prompt=lambda x: f"expand: {x}"),
    ]
    result = run_chain(steps, "topic: solar ovens", call_fn=stub)

    assert result.halted_at is None
    assert len(result.steps) == 2
    assert result.steps[0][0] == "outline"
    assert "topic: solar ovens" in result.steps[0][1]
    assert "expand:" in result.steps[1][1]  # second step received first step's output


def test_run_chain_gate_halts_early():
    def stub(system, prompt, temperature):
        return "too short"

    def reject_short(output):
        return "output too short" if len(output) < 20 else None

    steps = [
        Step(name="draft", system_prompt="draft", build_prompt=lambda x: x, gate=reject_short),
        Step(name="polish", system_prompt="polish", build_prompt=lambda x: x),
    ]
    result = run_chain(steps, "start", call_fn=stub)

    assert result.halted_at == "draft"
    assert result.halt_reason == "output too short"
    assert len(result.steps) == 1  # never reached "polish"


# ---------- routing ----------

def test_classify_matches_known_route():
    def stub(system, prompt, temperature):
        return "billing"

    routes = [
        Route(name="billing", description="payment questions", system_prompt="handle billing"),
        Route(name="technical", description="bug reports", system_prompt="handle technical"),
    ]
    assert classify("why was I charged twice?", routes, call_fn=stub) == "billing"


def test_classify_falls_back_to_first_route_on_unclear_answer():
    def stub(system, prompt, temperature):
        return "something unrelated entirely"

    routes = [
        Route(name="billing", description="payment questions", system_prompt="x"),
        Route(name="technical", description="bug reports", system_prompt="y"),
    ]
    assert classify("???", routes, call_fn=stub) == "billing"


def test_route_dispatches_to_matched_handler():
    calls = []

    def stub(system, prompt, temperature):
        calls.append(system)
        if "classifier" in system.lower():
            return "technical"
        return f"handled by: {system}"

    routes = [
        Route(name="billing", description="payment questions", system_prompt="billing-handler"),
        Route(name="technical", description="bug reports", system_prompt="technical-handler"),
    ]
    name, output = route("app crashes on launch", routes, call_fn=stub)
    assert name == "technical"
    assert output == "handled by: technical-handler"


# ---------- parallelization ----------

def test_section_preserves_input_order():
    def stub(system, prompt, temperature):
        return f"done:{prompt}"

    results = section(["a", "b", "c"], "system", call_fn=stub)
    assert results == ["done:a", "done:b", "done:c"]


def test_vote_returns_majority_answer():
    responses = iter(["yes", "no", "yes", "yes", "no"])

    def stub(system, prompt, temperature):
        return next(responses)

    winner, attempts = vote("is this safe?", "system", call_fn=stub, n=5)
    assert winner == "yes"
    assert len(attempts) == 5


# ---------- orchestrator_workers ----------

def test_orchestrate_parses_json_plan_and_synthesizes():
    def stub(system, prompt, temperature):
        if "JSON array" in system:
            return '["subtask A", "subtask B"]'
        if system == "worker":
            return f"result for: {prompt}"
        return "final synthesis"

    result = orchestrate("build a feature", worker_system_prompt="worker", call_fn=stub)
    assert result.subtasks == ["subtask A", "subtask B"]
    assert len(result.worker_outputs) == 2
    assert result.final == "final synthesis"
    assert result.plan_fallback is False


def test_orchestrate_falls_back_when_plan_is_not_json():
    def stub(system, prompt, temperature):
        if "JSON array" in system:
            return "not json at all"
        return "some output"

    result = orchestrate("a vague task", worker_system_prompt="worker", call_fn=stub)
    assert result.plan_fallback is True
    assert result.subtasks == ["a vague task"]
    assert len(result.worker_outputs) == 1


# ---------- evaluator_optimizer ----------

def test_refine_accepts_on_first_try():
    def stub(system, prompt, temperature):
        if "criteria" in system:
            return "ACCEPT"
        return "a fine first draft"

    result = refine("write a haiku", "must be exactly 3 lines", call_fn=stub)
    assert result.accepted is True
    assert result.iterations == 1
    assert result.output == "a fine first draft"
    assert result.history == []


def test_refine_loops_until_max_iterations_then_returns_last_attempt():
    call_count = {"n": 0}

    def stub(system, prompt, temperature):
        if "criteria" in system:
            return "needs more detail"  # never accepts
        call_count["n"] += 1
        return f"attempt {call_count['n']}"

    result = refine("summarize this", "must be detailed", call_fn=stub, max_iterations=3)
    assert result.accepted is False
    assert result.iterations == 3
    assert len(result.history) == 3
    assert result.output == "attempt 3"  # last attempt before giving up


if __name__ == "__main__":
    test_run_chain_passes_output_forward_through_steps()
    test_run_chain_gate_halts_early()
    test_classify_matches_known_route()
    test_classify_falls_back_to_first_route_on_unclear_answer()
    test_route_dispatches_to_matched_handler()
    test_section_preserves_input_order()
    test_vote_returns_majority_answer()
    test_orchestrate_parses_json_plan_and_synthesizes()
    test_orchestrate_falls_back_when_plan_is_not_json()
    test_refine_accepts_on_first_try()
    test_refine_loops_until_max_iterations_then_returns_last_attempt()
    print("All tests passed.")
