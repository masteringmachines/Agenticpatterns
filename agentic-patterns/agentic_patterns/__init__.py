"""
agentic_patterns
=================

Small, composable reference implementations of the patterns from
Anthropic's "Building Effective Agents" (Dec 2024):
https://www.anthropic.com/research/building-effective-agents

The article's central claim is that the most successful agent systems
in production use simple, composable patterns rather than heavyweight
frameworks — often just a few lines of code against an LLM API. This
package takes that literally: five workflow patterns plus the
"augmented LLM" building block they're all composed from, each a small
module with no framework underneath it.

    llm                  — the augmented LLM building block (llm.py)
    prompt_chaining      — Workflow: Prompt chaining
    routing              — Workflow: Routing
    parallelization      — Workflow: Parallelization (sectioning + voting)
    orchestrator_workers — Workflow: Orchestrator-workers
    evaluator_optimizer  — Workflow: Evaluator-optimizer

Every function takes a `call_fn: (system, user, temperature) -> str`
(see llm.py), so every pattern is unit-testable with a plain stub and
provider-agnostic by construction.
"""

from .llm import CallFn, default_call
from .prompt_chaining import Step, ChainResult, run_chain
from .routing import Route, classify, route
from .parallelization import section, vote
from .orchestrator_workers import OrchestrationResult, orchestrate
from .evaluator_optimizer import RefinementResult, refine

__all__ = [
    "CallFn",
    "default_call",
    "Step",
    "ChainResult",
    "run_chain",
    "Route",
    "classify",
    "route",
    "section",
    "vote",
    "OrchestrationResult",
    "orchestrate",
    "RefinementResult",
    "refine",
]

__version__ = "0.1.0"
