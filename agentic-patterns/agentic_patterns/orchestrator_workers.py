"""
orchestrator_workers.py — Workflow: Orchestrator-workers.

A central "orchestrator" call looks at the task and decides what
subtasks are needed — unlike parallelization's sectioning, the subtasks
aren't fixed ahead of time; the orchestrator determines them from the
specific input. Each subtask goes to a "worker" call, and a final call
synthesizes every worker's output into one result.

Good fit: tasks where you can't predict the subtasks in advance — the
article's example is a coding change where the number of files touched,
and how each needs to change, depends entirely on the task description.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field

from .llm import CallFn, default_call

ORCHESTRATOR_SYSTEM = (
    "You break a task down into a small list of independent subtasks for "
    "worker agents to complete. Respond with ONLY a JSON array of short "
    "subtask description strings — no other text, no markdown fences."
)

SYNTHESIS_SYSTEM = (
    "You combine several workers' outputs into one coherent final result "
    "for the original task."
)


@dataclass
class OrchestrationResult:
    subtasks: list[str]
    worker_outputs: list[str] = field(default_factory=list)
    final: str = ""
    plan_fallback: bool = False  # True if the orchestrator's plan wasn't valid JSON


def orchestrate(
    task: str,
    worker_system_prompt: str,
    call_fn: CallFn = default_call,
) -> OrchestrationResult:
    plan_text = call_fn(ORCHESTRATOR_SYSTEM, task, 0.3)

    plan_fallback = False
    try:
        subtasks = json.loads(plan_text)
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError("plan was not a non-empty JSON array")
    except (json.JSONDecodeError, ValueError):
        # A worse plan (treat the whole task as one subtask) beats no
        # plan at all if the orchestrator's output wasn't parseable.
        subtasks = [task]
        plan_fallback = True

    worker_outputs = [call_fn(worker_system_prompt, subtask, 0.5) for subtask in subtasks]

    combined = "\n\n".join(f"[{i + 1}] {out}" for i, out in enumerate(worker_outputs))
    synthesis_prompt = (
        f"Original task: {task}\n\nWorker outputs:\n{combined}\n\n"
        "Synthesize these into the final result for the original task."
    )
    final = call_fn(SYNTHESIS_SYSTEM, synthesis_prompt, 0.3)

    return OrchestrationResult(
        subtasks=subtasks, worker_outputs=worker_outputs, final=final, plan_fallback=plan_fallback
    )
