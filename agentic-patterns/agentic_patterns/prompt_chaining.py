"""
prompt_chaining.py — Workflow: Prompt chaining.

Decomposes a task into a sequence of LLM calls, where each call's
output feeds the next. An optional "gate" on any step can inspect its
output and halt the chain early — e.g. reject an outline that's missing
a required section before spending a call writing the full draft from it.

Good fit: a task that decomposes cleanly into fixed subtasks, trading
some latency for higher accuracy by giving each call an easier, more
focused job. Not a fit for tasks whose structure you can't predict in
advance — see orchestrator_workers.py for that case instead.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from .llm import CallFn, default_call


@dataclass
class Step:
    name: str
    system_prompt: str
    build_prompt: Callable[[str], str]  # (previous step's output) -> this step's user prompt
    temperature: float = 0.5
    # Optional check on this step's output. Return an error string to
    # halt the chain there, or None to continue to the next step.
    gate: Optional[Callable[[str], Optional[str]]] = None


@dataclass
class ChainResult:
    output: str
    steps: list[tuple[str, str]] = field(default_factory=list)  # (step name, output)
    halted_at: Optional[str] = None
    halt_reason: Optional[str] = None


def run_chain(steps: list[Step], initial_input: str, call_fn: CallFn = default_call) -> ChainResult:
    result = ChainResult(output=initial_input)
    current = initial_input

    for step in steps:
        prompt = step.build_prompt(current)
        output = call_fn(step.system_prompt, prompt, step.temperature)
        result.steps.append((step.name, output))

        if step.gate:
            error = step.gate(output)
            if error:
                result.halted_at = step.name
                result.halt_reason = error
                result.output = output
                return result

        current = output

    result.output = current
    return result
