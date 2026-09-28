"""Routing eval: run the real graph on labelled requests and score which specialists it called.

Uses Gemini (needs GOOGLE_API_KEY) and the live Nominatim/Open-Meteo APIs, so it is
not part of CI. Cases run one at a time to stay within API rate limits.

    cd backend && python -m evals.run_routing_eval [--cases evals/routing_cases.jsonl] [--limit N]

Per case, precision = |expected ∩ called| / |called| and recall = |expected ∩ called| /
|expected|, with an empty denominator counting as 1.0 (calling nothing when nothing is
expected is correct).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from app.agents.common import SPECIALISTS
from app.config import Settings, get_settings
from app.conversation import build_conversation
from app.graph.workflow import create_workflow, initial_state, run_config
from app.schemas import ChatTurn
from app.tools.http import create_http_client

DEFAULT_CASES = Path(__file__).with_name("routing_cases.jsonl")


@dataclass(frozen=True)
class Case:
    id: str
    message: str
    expected: frozenset[str]
    history: tuple[ChatTurn, ...] = ()


@dataclass(frozen=True)
class CaseResult:
    case: Case
    called: frozenset[str]
    precision: float
    recall: float
    steps: int
    latency_s: float
    finish_reason: str
    error: str | None = None


def precision_recall(expected: frozenset[str], called: frozenset[str]) -> tuple[float, float]:
    hits = len(expected & called)
    precision = hits / len(called) if called else 1.0
    recall = hits / len(expected) if expected else 1.0
    return precision, recall


def load_cases(path: Path) -> list[Case]:
    cases = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        expected = frozenset(raw["expected"])
        unknown = expected - set(SPECIALISTS)
        if unknown:
            raise ValueError(f"{path}:{line_no}: unknown specialists {sorted(unknown)}")
        history = tuple(ChatTurn.model_validate(turn) for turn in raw.get("history", []))
        cases.append(Case(id=raw["id"], message=raw["message"], expected=expected, history=history))
    return cases


async def run_case(graph: CompiledStateGraph, case: Case, settings: Settings) -> CaseResult:
    messages = build_conversation(case.history, case.message, settings.max_history_turns)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.graph_timeout_s):
            state = await graph.ainvoke(
                initial_state(messages, settings.max_steps), config=run_config(settings.max_steps)
            )
    except Exception as exc:  # report and keep going: one failure should not end the eval
        return CaseResult(
            case=case,
            called=frozenset(),
            precision=0.0,
            recall=0.0,
            steps=0,
            latency_s=time.perf_counter() - started,
            finish_reason="error",
            error=f"{type(exc).__name__}: {exc}"[:200],
        )
    latency_s = time.perf_counter() - started
    called = frozenset(event["node"] for event in state["trace"] if event["node"] in SPECIALISTS)
    precision, recall = precision_recall(case.expected, called)
    return CaseResult(
        case=case,
        called=called,
        precision=precision,
        recall=recall,
        steps=state.get("step_count", 0),
        latency_s=latency_s,
        finish_reason=state.get("finish_reason", "?"),
    )


def _names(names: frozenset[str]) -> str:
    return ",".join(sorted(names)) or "-"


def print_report(results: list[CaseResult], settings: Settings) -> None:
    header = f"{'case':<30} {'P':>5} {'R':>5} {'steps':>5} {'sec':>6}  expected -> called"
    print(header)
    print("-" * len(header))
    for r in results:
        line = (
            f"{r.case.id:<30} {r.precision:>5.2f} {r.recall:>5.2f} {r.steps:>5} "
            f"{r.latency_s:>6.1f}  {_names(r.case.expected)} -> {_names(r.called)}"
        )
        if r.finish_reason != "supervisor":
            line += f"  [{r.finish_reason}]"
        print(line)
        if r.error:
            print(f"{'':<30} error: {r.error}")

    ok = [r for r in results if r.error is None]
    print()
    print(f"model:            {settings.gemini_model}")
    print(f"date (UTC):       {datetime.now(UTC).isoformat(timespec='seconds')}")
    print(f"cases:            {len(results)} ({len(results) - len(ok)} errors)")
    if ok:
        exact = sum(r.called == r.case.expected for r in ok)
        print(f"mean precision:   {statistics.mean(r.precision for r in ok):.3f}")
        print(f"mean recall:      {statistics.mean(r.recall for r in ok):.3f}")
        print(f"exact set match:  {exact}/{len(ok)}")
        print(f"avg steps:        {statistics.mean(r.steps for r in ok):.2f}")
        print(f"avg latency (s):  {statistics.mean(r.latency_s for r in ok):.1f}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    args = parser.parse_args()

    load_dotenv()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("GOOGLE_API_KEY is not set (see backend/.env.example).", file=sys.stderr)
        return 1

    settings = get_settings()
    cases = load_cases(args.cases)[: args.limit]
    async with create_http_client(settings.http_user_agent) as http:
        graph = create_workflow(http, settings)
        results = [await run_case(graph, case, settings) for case in cases]
    print_report(results, settings)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
