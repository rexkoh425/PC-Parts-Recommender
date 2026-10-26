"""Resumable LLM weak-label pipeline for component-ranking candidate pairs.

Input JSONL rows contain ``query_id``, ``query_group_id``, a ranking ``context``,
an unscored ``candidate`` feature object, and an authorized ``judge_input`` object
with query requirements, structured product facts, stock state, and source evidence.
Retrieval scores and ranks are retained for training but never sent to the judge.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training._common import sha256_file

PROMPT_VERSION = "pcbr-llm-relevance-rubric-v2"
RUBRIC = """
Grade relevance from 0 to 4 for this single component request.
4: excellent fit for the stated workload and all known requirements.
3: strong fit; minor trade-off only.
2: plausible fit, with a meaningful limitation or important unknown.
1: weak fit; only loosely related or unsuitable for much of the request.
0: irrelevant, wrong category, or contradicts a stated hard requirement.
Use only supplied facts and authorized source evidence. Never infer unknown specs,
stock, price, compatibility, or benchmark performance. A hard requirement violation
cannot receive grade 3 or 4. Retrieval scores, ranks, and desired outcomes are hidden.
Return one JSON object with exactly these fields: relevance_grade (integer 0-4),
confidence (number 0-1), explanation (nonempty string), and
missing_or_unsupported_information (array of strings). Do not add markdown.
""".strip()
_ALLOWED_GRADES = {0, 1, 2, 3, 4}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_BLINDED_FIELDS = {
    "retrieval_scores",
    "retrieval_rank",
    "candidate_rank",
    "bm25_score",
    "vector_similarity",
    "rrf_score",
    "rank_position",
    "rank",
    "position",
    "model_identity",
    "desired_outcome",
}


def _blind(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _blind(item)
            for key, item in value.items()
            if str(key).casefold() not in _BLINDED_FIELDS
        }
    if isinstance(value, list):
        return [_blind(item) for item in value]
    return value


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _matches(actual: Any, operator: str, expected: Any) -> bool | None:
    if actual is None or actual == "" or actual == [] or actual == {}:
        return None
    try:
        if operator in {"eq", "equals"}:
            return str(actual).casefold() == str(expected).casefold()
        if operator in {"ge", "gte", "at_least"}:
            return actual >= expected
        if operator in {"le", "lte", "at_most"}:
            return actual <= expected
        if operator == "gt":
            return actual > expected
        if operator == "lt":
            return actual < expected
        if operator == "one_of":
            return str(actual).casefold() in {str(item).casefold() for item in expected}
        if operator == "contains":
            return expected in actual
        if operator == "contains_all":
            return all(item in actual for item in expected)
        if operator == "not_null":
            return actual != "" and actual != []
    except (TypeError, ValueError):
        return False
    raise ValueError(f"unsupported hard-requirement operator: {operator}")


def hard_requirement_outcome(task: Mapping[str, Any]) -> tuple[bool, bool, list[str]]:
    """Return (violated, unknown, explanations) using only explicit hard constraints."""

    judge_input = task["judge_input"]
    specs = judge_input.get("specifications", {})
    violated = False
    unknown = False
    details: list[str] = []
    requirements = judge_input.get("requirements", [])
    if not isinstance(requirements, list):
        raise ValueError("judge_input.requirements must be a JSON array")
    for requirement in requirements:
        if not isinstance(requirement, Mapping) or not requirement.get("hard", False):
            continue
        name = str(requirement.get("name", requirement.get("field", "requirement")))
        field = str(requirement.get("field", name))
        actual = _nested(specs, field)
        outcome = _matches(actual, str(requirement.get("operator", "eq")), requirement.get("value"))
        if outcome is False:
            violated = True
            details.append(f"hard requirement failed: {name}")
        elif outcome is None:
            unknown = True
            details.append(f"hard requirement unknown: {name}")
    return violated, unknown, details


def _validate_response(payload: object, *, model_id: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("judge response must be a JSON object")
    grade = payload.get("relevance_grade")
    confidence = payload.get("confidence")
    explanation = payload.get("explanation")
    missing = payload.get("missing_or_unsupported_information")
    if type(grade) is not int or grade not in _ALLOWED_GRADES:
        raise ValueError("relevance_grade must be an integer from 0 through 4")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be numeric from 0 through 1")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation must be a non-empty string")
    if not isinstance(missing, list) or any(not isinstance(item, str) for item in missing):
        raise ValueError("missing_or_unsupported_information must be a string array")
    return {
        "relevance_grade": grade,
        "confidence": float(confidence),
        "explanation": explanation.strip(),
        "missing_or_unsupported_information": missing,
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
    }


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    base_url: str
    api_key: str
    model_id: str
    timeout_seconds: float = 90.0
    max_retries: int = 4
    concurrency: int = 2
    requests_per_second: float = 1.0
    max_consecutive_failures: int = 10
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    @classmethod
    def from_environment(cls, **overrides: Any) -> JudgeConfig:
        base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        return cls(
            base_url=overrides.pop("base_url", base),
            api_key=overrides.pop("api_key", os.environ.get("OPENAI_API_KEY", "")),
            model_id=overrides.pop(
                "model_id",
                os.environ.get("LLM_JUDGE_MODEL", os.environ.get("CLIPROXY_CODEX_MODEL", "")),
            ),
            **overrides,
        )

    def validate(self) -> None:
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL or --base-url is required")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required; credentials are never written to artifacts"
            )
        if not self.model_id:
            raise ValueError("LLM_JUDGE_MODEL or CLIPROXY_CODEX_MODEL is required")
        if (
            self.concurrency < 1
            or self.requests_per_second <= 0
            or self.max_retries < 0
            or self.max_consecutive_failures < 1
        ):
            raise ValueError("invalid concurrency, request-rate, or retry configuration")


class _RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.next_time = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if delay:
            time.sleep(delay)


def balance_pending_by_group(
    pending: list[dict[str, Any]], snapshot_path: Path
) -> list[dict[str, Any]]:
    """Round-robin complete intent groups across frozen category/split strata."""
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    by_query = {row["query_id"]: row for row in snapshot}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strata: dict[str, tuple[str, str]] = {}
    for task in pending:
        row = by_query.get(task["query_id"])
        if row is None or row["query_group_id"] != task.get("query_group_id", task["query_id"]):
            raise ValueError("schedule snapshot does not match pending query groups")
        group = row["query_group_id"]
        key = (row["split"], row["category"])
        if group in strata and strata[group] != key:
            raise ValueError("schedule group spans categories or splits")
        strata[group] = key
        grouped[group].append(task)
    buckets: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    for group in sorted(grouped):
        buckets[strata[group]].append(group)
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                group = buckets[key].popleft()
                ordered.extend(sorted(grouped[group], key=lambda task: task["task_id"]))
    return ordered


class OpenAICompatibleJudge:
    """Small standard-library client for an OpenAI-compatible chat endpoint."""

    def __init__(self, config: JudgeConfig) -> None:
        config.validate()
        self.config = config
        self.rate_limiter = _RateLimiter(config.requests_per_second)

    @property
    def chat_completions_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return base + "/chat/completions"

    def judge(self, task: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        judge_input = task.get("judge_input")
        if not isinstance(judge_input, Mapping):
            raise ValueError("task must include a judge_input JSON object")
        product = judge_input.get("product")
        if not isinstance(product, Mapping):
            raise ValueError("judge_input.product must contain actual product evidence")
        if (
            str(product.get("category", "")).casefold()
            != str(judge_input.get("category", "")).casefold()
        ):
            raise ValueError("candidate category does not match query category")
        body = {
            "model": self.config.model_id,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": RUBRIC},
                {
                    "role": "user",
                    "content": json.dumps(_blind(judge_input), sort_keys=True, ensure_ascii=False),
                },
            ],
        }
        request = urllib.request.Request(
            self.chat_completions_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.config.max_retries + 1):
            self.rate_limiter.wait()
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    payload = json.loads(response.read())
                content = payload["choices"][0]["message"]["content"]
                judgment = _validate_response(json.loads(content), model_id=self.config.model_id)
                usage = payload.get("usage", {})
                return judgment, {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                }
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                retryable = True
                if isinstance(error, urllib.error.HTTPError):
                    retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt >= self.config.max_retries:
                    raise RuntimeError(
                        f"judge request failed for task {task.get('task_id')}: {error}"
                    ) from error
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise AssertionError("unreachable retry loop")


def judge_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    config: JudgeConfig,
    schedule_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Judge all pairs, resuming exact task hashes from the checkpoint JSONL."""

    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "judgments.checkpoint.jsonl"
    failures_path = output_dir / "failures.jsonl"
    tasks: list[dict[str, Any]] = []
    for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("task_id"):
                raise ValueError(f"input row {line_no} requires task_id")
            row["_task_hash"] = _canonical_hash(
                {
                    "task": {key: value for key, value in row.items() if key != "_task_hash"},
                    "model_id": config.model_id,
                    "prompt_version": PROMPT_VERSION,
                    "rubric_sha256": _canonical_hash(RUBRIC),
                }
            )
            tasks.append(row)
    ids = [row["task_id"] for row in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task_id values must be unique")

    done: dict[str, dict[str, Any]] = {}
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = json.loads(line)
                done[result["task_id"]] = result
    pending = [
        row for row in tasks if done.get(row["task_id"], {}).get("task_hash") != row["_task_hash"]
    ]
    if schedule_snapshot is not None:
        pending = balance_pending_by_group(pending, schedule_snapshot)
    client = OpenAICompatibleJudge(config)
    write_lock = threading.Lock()

    def run(task: dict[str, Any]) -> dict[str, Any]:
        judgment, usage = client.judge(task)
        raw_grade = judgment["relevance_grade"]
        violated, unknown, deterministic_details = hard_requirement_outcome(task)
        if violated:
            judgment["relevance_grade"] = 0
        elif unknown:
            judgment["relevance_grade"] = min(2, judgment["relevance_grade"])
        judgment["missing_or_unsupported_information"] = list(
            dict.fromkeys(judgment["missing_or_unsupported_information"] + deterministic_details)
        )
        return {
            "task_id": task["task_id"],
            "task_hash": task["_task_hash"],
            "query_id": task["query_id"],
            "query_group_id": task.get("query_group_id", task["query_id"]),
            "product_id": task["candidate"].get("product_id"),
            "candidate": task["candidate"],
            "context": task["context"],
            "judgment": judgment,
            "judge_raw_relevance_grade": raw_grade,
            "hard_requirement_violated": violated,
            "hard_requirement_unknown": unknown,
            "usage": usage,
            "label_source": "llm_weak",
            "authorized_evidence_sha256": _canonical_hash(
                task["judge_input"].get("source_evidence", [])
            ),
        }

    consecutive_failures = 0
    failed_this_run = 0
    attempted_this_run = 0
    circuit_tripped = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        task_iterator = iter(pending)
        inflight: dict[concurrent.futures.Future, dict[str, Any]] = {}

        def fill() -> None:
            while not circuit_tripped and len(inflight) < config.concurrency:
                task = next(task_iterator, None)
                if task is None:
                    break
                inflight[pool.submit(run, task)] = task

        fill()
        while inflight:
            completed, _ = concurrent.futures.wait(
                inflight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                task = inflight.pop(future)
                attempted_this_run += 1
                try:
                    result = future.result()
                except Exception as error:
                    failed_this_run += 1
                    consecutive_failures += 1
                    cause = error.__cause__
                    status = cause.code if isinstance(cause, urllib.error.HTTPError) else None
                    # Do not persist provider response bodies, URLs, or credentials.
                    with failures_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "task_id": task["task_id"],
                                    "task_hash": task["_task_hash"],
                                    "error_type": type(error).__name__,
                                    "http_status": status,
                                }
                            )
                            + "\n"
                        )
                    if consecutive_failures >= config.max_consecutive_failures:
                        circuit_tripped = True
                    continue
                consecutive_failures = 0
                with write_lock, checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
                    handle.flush()
                done[result["task_id"]] = result
            fill()

    results = [
        done[task["task_id"]]
        for task in tasks
        if done.get(task["task_id"], {}).get("task_hash") == task["_task_hash"]
    ]
    expected_counts = Counter(task["query_id"] for task in tasks)
    completed_counts = Counter(result["query_id"] for result in results)
    complete_queries = {
        query_id
        for query_id, count in expected_counts.items()
        if completed_counts[query_id] == count
    }
    grouped: dict[str, dict[str, Any]] = {}
    for result in results:
        query_id = result["query_id"]
        if query_id not in complete_queries:
            continue
        group = grouped.setdefault(
            query_id,
            {
                "context": dict(result["context"]),
                "candidates": [],
                "query_group_id": result["query_group_id"],
            },
        )
        candidate = dict(result["candidate"])
        candidate["relevance_grade"] = result["judgment"]["relevance_grade"]
        group["candidates"].append(candidate)
    ranking_path = output_dir / "ranking_dataset.llm_weak.jsonl"
    ranking_rows = []
    for query_id in sorted(grouped):
        group = grouped[query_id]
        group["context"].setdefault("query_group_id", group["query_group_id"])
        group["candidates"].sort(key=lambda candidate: candidate["product_id"])
        ranking_rows.append({"context": group["context"], "candidates": group["candidates"]})
    # A partial dataset would change a later seed-based split and leak held-out
    # intents into training on resume. Publish training rows only after completion.
    if len(results) != len(tasks):
        ranking_rows = []
    ranking_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in ranking_rows),
        encoding="utf-8",
    )
    grades = [row["judgment"]["relevance_grade"] for row in results]
    tokens_in = sum(row["usage"]["prompt_tokens"] for row in results)
    tokens_out = sum(row["usage"]["completion_tokens"] for row in results)
    category_grades: dict[str, Counter[int]] = {}
    for result in results:
        category = result["candidate"]["category"]
        category_grades.setdefault(category, Counter())[result["judgment"]["relevance_grade"]] += 1
    summary = {
        "schema_version": "pc-build-recommender.llm-weak-judgments.v1",
        "label_source": "llm_weak",
        "eligible_for_production_or_human_quality_claims": False,
        "model_id": config.model_id,
        "prompt_version": PROMPT_VERSION,
        "rubric_sha256": _canonical_hash(RUBRIC),
        "input_sha256": sha256_file(input_path),
        "judgments_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
        "ranking_dataset_sha256": sha256_file(ranking_path),
        "task_count": len(results),
        "input_task_count": len(tasks),
        "unresolved_task_count": len(tasks) - len(results),
        "attempted_this_run": attempted_this_run,
        "failed_this_run": failed_this_run,
        "circuit_breaker_tripped": circuit_tripped,
        "not_attempted_this_run": len(pending) - attempted_this_run,
        "schedule_snapshot_sha256": sha256_file(schedule_snapshot) if schedule_snapshot else None,
        "training_dataset_ready": len(results) == len(tasks) and bool(tasks),
        "incomplete_query_count": len({task["query_id"] for task in tasks} - complete_queries),
        "query_count": len(grouped),
        "query_group_count": len({row["query_group_id"] for row in results}),
        "grade_counts": {str(grade): grades.count(grade) for grade in sorted(_ALLOWED_GRADES)},
        "grade_counts_by_category": {
            category: {str(grade): counts[grade] for grade in sorted(_ALLOWED_GRADES)}
            for category, counts in sorted(category_grades.items())
        },
        "confidence_buckets": {
            "below_0.5": sum(row["judgment"]["confidence"] < 0.5 for row in results),
            "0.5_to_below_0.8": sum(0.5 <= row["judgment"]["confidence"] < 0.8 for row in results),
            "0.8_to_1.0": sum(row["judgment"]["confidence"] >= 0.8 for row in results),
        },
        "missing_evidence_count": sum(
            bool(row["judgment"]["missing_or_unsupported_information"]) for row in results
        ),
        "raw_hard_requirement_contradiction_count": sum(
            row["hard_requirement_violated"] and row.get("judge_raw_relevance_grade", 0) > 0
            for row in results
        ),
        "raw_high_grade_hard_contradiction_count": sum(
            row["hard_requirement_violated"] and row.get("judge_raw_relevance_grade", 0) >= 3
            for row in results
        ),
        "raw_grade_recorded_count": sum("judge_raw_relevance_grade" in row for row in results),
        "final_hard_requirement_contradiction_count": sum(
            row["hard_requirement_violated"] and row["judgment"]["relevance_grade"] != 0
            for row in results
        ),
        "low_confidence_count": sum(row["judgment"]["confidence"] < 0.5 for row in results),
        "hard_requirement_violation_count": sum(
            row["hard_requirement_violated"] for row in results
        ),
        "hard_requirement_unknown_count": sum(row["hard_requirement_unknown"] for row in results),
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "estimated_cost_usd": (
            (
                tokens_in * config.input_cost_per_million
                + tokens_out * config.output_cost_per_million
            )
            / 1_000_000
            if config.input_cost_per_million or config.output_cost_per_million
            else None
        ),
        "checkpoint": checkpoint.name,
        "ranking_dataset": ranking_path.name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="query-product judge tasks JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-consecutive-failures", type=int, default=10)
    parser.add_argument("--schedule-snapshot", type=Path)
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    args = parser.parse_args()
    config = JudgeConfig.from_environment(
        model_id=args.model
        or os.environ.get("LLM_JUDGE_MODEL", os.environ.get("CLIPROXY_CODEX_MODEL", "")),
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL", ""),
        concurrency=args.concurrency,
        requests_per_second=args.requests_per_second,
        max_retries=args.max_retries,
        max_consecutive_failures=args.max_consecutive_failures,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    summary = judge_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        config=config,
        schedule_snapshot=args.schedule_snapshot,
    )
    print(json.dumps(summary, indent=2))
    return int(summary["unresolved_task_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
