"""
evaluator_optimizer.py — Workflow: Evaluator-optimizer.

One call generates a response; a separate call evaluates it against
explicit criteria and either accepts it or returns specific feedback.
On feedback, the generator retries with that feedback in view — looping
until accepted or a max-iteration safety valve is hit.

Good fit, per the article: cases where a human's feedback demonstrably
improves the response, and an LLM can produce feedback of similar
quality — e.g. literary translation, where nuances a first pass misses
are exactly the kind of thing a critical second read catches.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .llm import CallFn, default_call

GENERATOR_SYSTEM = "You produce a response to the task, incorporating any feedback given."
EVALUATOR_SYSTEM_TEMPLATE = (
    "You evaluate a response against this criteria: {criteria}\n"
    "If it fully meets the criteria, respond with exactly: ACCEPT\n"
    "Otherwise, respond with specific, actionable feedback for improvement — "
    "no praise, no preamble, just what to fix."
)


@dataclass
class RefinementResult:
    output: str
    accepted: bool
    iterations: int
    history: list[tuple[str, str]] = field(default_factory=list)  # (attempt, feedback)


def refine(
    task: str,
    criteria: str,
    call_fn: CallFn = default_call,
    max_iterations: int = 3,
) -> RefinementResult:
    evaluator_system = EVALUATOR_SYSTEM_TEMPLATE.format(criteria=criteria)
    prompt = task
    history: list[tuple[str, str]] = []

    for i in range(max_iterations):
        output = call_fn(GENERATOR_SYSTEM, prompt, 0.6)
        feedback = call_fn(evaluator_system, output, 0.0).strip()

        if feedback.upper().startswith("ACCEPT"):
            return RefinementResult(output=output, accepted=True, iterations=i + 1, history=history)

        history.append((output, feedback))
        prompt = (
            f"Original task: {task}\n\nYour previous attempt:\n{output}\n\n"
            f"Feedback to address:\n{feedback}"
        )

    # Ran out of iterations without acceptance — return the last attempt
    # rather than raising, so the caller can still decide what to do
    # with a "close but not accepted" result.
    last_output = history[-1][0] if history else ""
    return RefinementResult(output=last_output, accepted=False, iterations=max_iterations, history=history)
