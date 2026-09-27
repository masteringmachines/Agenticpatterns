"""
parallelization.py — Workflow: Parallelization.

Two variations, per the article:
  - sectioning (`section`): split a task into independent subtasks, run
    them concurrently, and combine the results programmatically.
  - voting (`vote`): run the SAME task multiple times — usually at a
    nonzero temperature for diversity — and aggregate the independent
    attempts into one higher-confidence answer.

Good fit: subtasks that are genuinely independent of each other
(sectioning), or tasks where several separate attempts and a vote catch
more issues than one careful attempt would — e.g. flagging a piece of
code for security problems from a few different angles.

Uses a plain ThreadPoolExecutor rather than asyncio: call_fn is a
synchronous function (see llm.py), and threads are enough concurrency
for a handful of I/O-bound API calls without needing an async rewrite
of every pattern in this package.
"""

from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .llm import CallFn, default_call


def section(
    tasks: list[str],
    system_prompt: str,
    call_fn: CallFn = default_call,
    max_workers: int = 5,
) -> list[str]:
    """Runs one call per task concurrently. Returns results in the same
    order as `tasks`, regardless of which finishes first."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda t: call_fn(system_prompt, t, 0.5), tasks))


def vote(
    prompt: str,
    system_prompt: str,
    call_fn: CallFn = default_call,
    n: int = 5,
    temperature: float = 0.7,
    max_workers: int = 5,
) -> tuple[str, list[str]]:
    """Runs the same prompt `n` times concurrently and returns the most
    common exact response plus every individual attempt.

    Best suited to answers with a small set of likely values (a
    classification, a yes/no, a short label). For open-ended free text
    where responses rarely match verbatim, use the returned `attempts`
    list with your own aggregation (e.g. a follow-up call that
    summarizes points of agreement) instead of the majority pick.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        attempts = list(pool.map(lambda _: call_fn(system_prompt, prompt, temperature), range(n)))
    winner, _ = Counter(attempts).most_common(1)[0]
    return winner, attempts
