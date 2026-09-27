# agentic-patterns

Small, composable reference implementations of the patterns from
Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
(Dec 2024). Five workflow patterns, plus the "augmented LLM" building
block they're all composed from — one small module each, no framework
underneath any of them.

The article's central claim, paraphrased: the most successful agent
systems in production don't come from heavyweight frameworks — they
come from a handful of simple patterns implemented directly against an
LLM API. This repo takes that literally and implements each pattern in
isolation, so you can see exactly what each one does without a
framework's abstraction in the way.

```bash
python examples/demo.py   # runs all five patterns, no API key needed
```

## The building block, and the five patterns

| Module | Article section | What it does |
|---|---|---|
| `llm.py` | *The augmented LLM* | Fixes the one shape (`system, user, temperature -> str`) every pattern below is built on, so each is testable with a stub and provider-agnostic. |
| `prompt_chaining.py` | *Workflow: Prompt chaining* | A sequence of calls, each fed the previous one's output. Any step can have a "gate" that halts the chain if its output fails a check. |
| `routing.py` | *Workflow: Routing* | Classifies an input, then dispatches to one of several specialized handlers with their own tuned prompts. |
| `parallelization.py` | *Workflow: Parallelization* | Two variations: `section()` splits a task into independent pieces run concurrently; `vote()` runs the same task several times and takes the majority answer. |
| `orchestrator_workers.py` | *Workflow: Orchestrator-workers* | A central call decides what subtasks are needed (unlike sectioning, not fixed in advance), hands each to a worker call, then synthesizes the results. |
| `evaluator_optimizer.py` | *Workflow: Evaluator-optimizer* | A generator call and a separate evaluator call loop — generate, critique against explicit criteria, regenerate with that feedback — until accepted or a max-iteration limit. |

Each module's docstring includes the article's own "when to use this"
guidance for that pattern, paraphrased, so you don't have to cross-
reference the original post while reading the code.

## Install

```bash
git clone https://github.com/YOUR-USERNAME/agentic-patterns.git
cd agentic-patterns
pip install -e .
export ANTHROPIC_API_KEY=sk-...
```

## Usage

Every pattern takes an optional `call_fn` — default it to a real model
call, or pass a stub for testing:

```python
from agentic_patterns import default_call, Route, route

routes = [
    Route("billing", "payment or refund questions", "You are a billing specialist."),
    Route("technical", "bug reports or how-to questions", "You are technical support."),
]
name, answer = route("I'd like a refund for my last order", routes, call_fn=default_call)
print(name, "->", answer)
```

```python
from agentic_patterns import section, default_call

# Sectioning: three independent subtasks, run concurrently
results = section(
    ["Summarize the intro", "Summarize the methods", "Summarize the results"],
    system_prompt="You write a one-sentence summary.",
    call_fn=default_call,
)
```

```python
from agentic_patterns import refine

result = refine(
    task="Write a one-paragraph pitch for a community tool library.",
    criteria="must include one concrete, specific example",
    max_iterations=3,
)
print(result.accepted, result.output)
```

See `examples/demo.py` for all five patterns run end to end with a
deterministic stub — no API key needed to see the control flow of each.

## Test (no API key needed)

Every pattern is tested with a plain stub `call_fn`, so the whole suite
runs without touching the network:

```bash
python tests/test_patterns.py
```

## Why this shape

The article is explicit that frameworks "often create extra layers of
abstraction that can obscure the underlying prompts and responses,
making them harder to debug," and recommends starting with direct API
calls before reaching for one. This repo is what that advice looks like
taken all the way: each pattern is a few dozen lines, readable start to
finish, with the actual prompts sent to the model always visible in the
code rather than hidden inside a class hierarchy.

That also means each pattern is trivially swappable:

- Change providers by replacing `default_call` in `llm.py` — every
  pattern only depends on its `(system, user, temperature) -> str` shape.
- Add tools or retrieval to the augmented LLM by changing what
  `default_call` does internally; no pattern module needs to know.
- Combine patterns freely — `orchestrate()`'s worker calls could
  themselves be a `run_chain()`, a `route()` could dispatch into a
  `refine()` loop, and so on. The article's own point is that these are
  composable building blocks, not a fixed pipeline.

## Design goals

- **Small.** Each pattern module is under 80 lines.
- **Composable, not a framework.** No base classes, no plugin system —
  plain functions and dataclasses you can read top to bottom.
- **Provider-agnostic.** One function (`default_call`) touches the
  network; everything else only depends on its input/output shape.
- **Testable without a network.** Every pattern's control flow (gates,
  fallbacks, majority votes, iteration limits) is covered by a stub-
  based test, independent of model quality.

## License

MIT — see [LICENSE](LICENSE). Not affiliated with or endorsed by
Anthropic; a reference implementation inspired by their published
article, not the article's own code.
