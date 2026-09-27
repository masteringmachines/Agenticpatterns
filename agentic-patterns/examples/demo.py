"""
demo.py — runs all five patterns end to end with a stub `call_fn`, so
you can see the control flow of each pattern with no API key and no
network call. Swap `stub_call` for `agentic_patterns.default_call` (and
export ANTHROPIC_API_KEY) to run them for real — nothing else changes.

    python examples/demo.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_patterns import (
    Step, run_chain,
    Route, route,
    section, vote,
    orchestrate,
    refine,
)


def stub_call(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """A deterministic fake model, just so this demo runs with zero
    setup. Replace with agentic_patterns.default_call for the real thing."""
    s = system_prompt.lower()
    if "classify" in s or "classifier" in s:
        if "refund" in user_prompt.lower():
            return "billing"
        return "technical"
    if "json array" in s:
        return '["research the topic", "draft an outline", "check for factual gaps"]'
    if "combine several workers" in s:
        return "Combined result: a short, well-sourced draft."
    if "criteria" in s:
        return "ACCEPT" if "final" in user_prompt.lower() else "add one more concrete example"
    if "generator" in s or "produce a response" in s:
        return "final draft with a concrete example included"
    return f"[stub response to: {user_prompt[:40]}...]"


def demo_prompt_chaining():
    print("\n=== Prompt chaining ===")
    steps = [
        Step("outline", "You write a bullet outline.", build_prompt=lambda x: x),
        Step("draft", "You expand an outline into full prose.", build_prompt=lambda x: f"Expand: {x}"),
    ]
    result = run_chain(steps, "Topic: composting for apartment dwellers", call_fn=stub_call)
    for name, output in result.steps:
        print(f"  [{name}] {output}")


def demo_routing():
    print("\n=== Routing ===")
    routes = [
        Route("billing", "payment or refund questions", "You are a billing specialist."),
        Route("technical", "bug reports or how-to questions", "You are technical support."),
    ]
    name, output = route("I'd like a refund for my last order", routes, call_fn=stub_call)
    print(f"  routed to: {name} -> {output}")


def demo_parallelization():
    print("\n=== Parallelization ===")
    sectioned = section(["water", "energy", "food"], "Give one solarpunk tip about:", call_fn=stub_call)
    for s in sectioned:
        print(f"  {s}")

    winner, attempts = vote("Is this code safe to merge?", "You review code for vulnerabilities.", call_fn=stub_call, n=5)
    print(f"  vote winner: {winner!r} (from {len(attempts)} attempts)")


def demo_orchestrator_workers():
    print("\n=== Orchestrator-workers ===")
    result = orchestrate("Write a short brief on urban beekeeping", worker_system_prompt="You are a research worker.", call_fn=stub_call)
    print(f"  subtasks: {result.subtasks}")
    print(f"  final: {result.final}")


def demo_evaluator_optimizer():
    print("\n=== Evaluator-optimizer ===")
    result = refine("Write a one-paragraph pitch for a tool library.", "must include a concrete example", call_fn=stub_call)
    print(f"  accepted: {result.accepted} after {result.iterations} iteration(s)")
    print(f"  output: {result.output}")


if __name__ == "__main__":
    demo_prompt_chaining()
    demo_routing()
    demo_parallelization()
    demo_orchestrator_workers()
    demo_evaluator_optimizer()
