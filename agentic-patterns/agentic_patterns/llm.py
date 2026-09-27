"""
llm.py — the "augmented LLM": the one building block every pattern in
this package is composed from.

Anthropic's "Building Effective Agents" describes the augmented LLM —
a model call enhanced with things like tools, retrieval, and memory —
as the foundation everything else (workflows, agents) is built on top
of. This module doesn't try to implement retrieval or memory generally;
it just fixes the *shape* every pattern below depends on, so each one
can be unit tested with a plain stub and swapped onto any provider.
"""

from __future__ import annotations
from typing import Callable

MODEL = "claude-sonnet-4-6"

# (system_prompt, user_prompt, temperature) -> response text.
# Every function in this package takes a `call_fn` of this exact shape.
CallFn = Callable[[str, str, float], str]


def default_call(system_prompt: str, user_prompt: str, temperature: float = 0.5) -> str:
    """The only function that touches the network. Swap this for a
    different provider's SDK, add tool use, add retrieval — none of the
    patterns in this package need to change, since they only depend on
    the (system, user, temperature) -> str shape above."""
    import anthropic  # imported lazily so the package works without it installed

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()
