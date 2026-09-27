"""
routing.py — Workflow: Routing.

Classifies an input, then dispatches it to one of several specialized
handlers, each with its own narrowly-tuned prompt — rather than one
prompt trying to handle every kind of input well. The article notes
classification can be an LLM call (as here), a smaller/cheaper model,
or plain rules; swap `classify()`'s body for any of those without
touching `route()`.

Good fit: distinct input categories that are each easier to handle with
a focused prompt, and where classification itself is reliable — e.g.
sorting support tickets into billing/technical/general before drafting
a reply for each.
"""

from __future__ import annotations
from dataclasses import dataclass

from .llm import CallFn, default_call


@dataclass
class Route:
    name: str
    description: str  # shown to the classifier so it can choose this route
    system_prompt: str  # used to actually handle the input once chosen


def classify(user_input: str, routes: list[Route], call_fn: CallFn = default_call) -> str:
    """Returns the chosen route's name. Falls back to the first route if
    the model's answer doesn't clearly match any known route — a
    reasonable default beats a crash on a slightly-off classification."""
    options = "\n".join(f"- {r.name}: {r.description}" for r in routes)
    prompt = (
        "Classify this input into exactly one category. Respond with only "
        f"the category name, nothing else.\n\nCategories:\n{options}\n\nInput: {user_input}"
    )
    answer = call_fn("You are a precise classifier.", prompt, 0.0).strip().lower()
    for r in routes:
        if r.name.lower() in answer:
            return r.name
    return routes[0].name


def route(user_input: str, routes: list[Route], call_fn: CallFn = default_call) -> tuple[str, str]:
    """Classifies the input, then runs the matched route's own prompt.
    Returns (chosen_route_name, output)."""
    chosen_name = classify(user_input, routes, call_fn)
    chosen = next(r for r in routes if r.name == chosen_name)
    output = call_fn(chosen.system_prompt, user_input, 0.5)
    return chosen_name, output
