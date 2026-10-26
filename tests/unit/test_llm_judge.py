from __future__ import annotations

import json

import pytest
from training.llm_judge import (
    PROMPT_VERSION,
    JudgeConfig,
    OpenAICompatibleJudge,
    _validate_response,
    balance_pending_by_group,
    hard_requirement_outcome,
    judge_dataset,
)


def _task(actual: int | None) -> dict:
    return {
        "judge_input": {
            "specifications": {"core_count": actual},
            "requirements": [
                {
                    "name": "at least 8 cores",
                    "field": "core_count",
                    "operator": "ge",
                    "value": 8,
                    "hard": True,
                }
            ],
        }
    }


def test_hard_requirements_detect_violation_and_unknown() -> None:
    assert hard_requirement_outcome(_task(6)) == (
        True,
        False,
        ["hard requirement failed: at least 8 cores"],
    )
    assert hard_requirement_outcome(_task(None)) == (
        False,
        True,
        ["hard requirement unknown: at least 8 cores"],
    )


def test_hard_requirements_cover_all_list_items_and_missing_fields() -> None:
    task = {
        "judge_input": {
            "specifications": {"sockets": ["AM4", "LGA 1700"]},
            "requirements": [
                {
                    "field": "sockets",
                    "operator": "contains_all",
                    "value": ["AM5", "LGA 1700"],
                    "hard": True,
                },
                {"field": "graphics", "operator": "not_null", "value": True, "hard": True},
            ],
        }
    }
    violated, unknown, _ = hard_requirement_outcome(task)
    assert violated and unknown


def test_strict_numeric_constraint_rejects_equality() -> None:
    task = {
        "judge_input": {
            "specifications": {"height": 70},
            "requirements": [{"field": "height", "operator": "lt", "value": 70, "hard": True}],
        }
    }
    assert hard_requirement_outcome(task)[0]


def test_empty_list_is_unknown_not_a_compatibility_violation() -> None:
    task = {
        "judge_input": {
            "specifications": {"sockets": []},
            "requirements": [
                {"field": "sockets", "operator": "contains", "value": "AM5", "hard": True}
            ],
        }
    }
    violated, unknown, _ = hard_requirement_outcome(task)
    assert not violated and unknown


def test_judge_response_schema_is_strict() -> None:
    response = _validate_response(
        {
            "relevance_grade": 3,
            "confidence": 0.8,
            "explanation": "Good known specification fit.",
            "missing_or_unsupported_information": ["stock"],
        },
        model_id="test-model",
    )
    assert response["prompt_version"] == PROMPT_VERSION
    assert response["model_id"] == "test-model"
    with pytest.raises(ValueError, match="0 through 4"):
        _validate_response(
            {
                "relevance_grade": 5,
                "confidence": 0.8,
                "explanation": "invalid",
                "missing_or_unsupported_information": [],
            },
            model_id="test-model",
        )


def test_judge_prompt_does_not_receive_retrieval_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return (
                b'{"choices":[{"message":{"content":"{\\"relevance_grade\\":3,'
                b'\\"confidence\\":0.8,\\"explanation\\":\\"Fits\\",'
                b'\\"missing_or_unsupported_information\\":[]} "}}],'
                b'"usage":{"prompt_tokens":10,"completion_tokens":5}}'
            )

    def _open(request, timeout):
        import json

        seen["body"] = json.loads(request.data)
        seen["url"] = request.full_url
        return _Response()

    monkeypatch.setattr("training.llm_judge.urllib.request.urlopen", _open)
    client = OpenAICompatibleJudge(
        JudgeConfig(
            base_url="http://localhost:1234",
            api_key="test-only",
            model_id="test-model",
            requests_per_second=1000,
        )
    )
    task = {
        "task_id": "q:p",
        "candidate": {"product_id": "p", "category": "gpu"},
        "judge_input": {
            "query_text": "quiet graphics card",
            "category": "gpu",
            "product": {"name": "GPU", "category": "gpu"},
            "specifications": {},
            "retrieval_scores": {"bm25_score": 999},
        },
    }
    judgment, usage = client.judge(task)
    assert judgment["relevance_grade"] == 3
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    prompt = " ".join(message["content"] for message in seen["body"]["messages"])
    assert "retrieval_scores" not in prompt


def test_failure_resume_excludes_incomplete_groups(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = []
    for query_id in ("q1", "q2", "q3"):
        for product_id in ("a", "b"):
            tasks.append(
                {
                    "task_id": f"{query_id}:{product_id}",
                    "query_id": query_id,
                    "query_group_id": query_id,
                    "context": {"query_id": query_id, "query_group_id": query_id},
                    "candidate": {"product_id": product_id, "category": "cpu"},
                    "judge_input": {
                        "product": {"name": product_id, "category": "cpu"},
                        "category": "cpu",
                        "requirements": [],
                        "specifications": {},
                    },
                }
            )
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    calls = []
    fail = {"q1:b"}

    def fake_judge(_self, task):
        calls.append(task["task_id"])
        if task["task_id"] in fail:
            raise RuntimeError("provider error with potentially sensitive body")
        return (
            {
                "relevance_grade": 3,
                "confidence": 0.8,
                "explanation": "test",
                "missing_or_unsupported_information": [],
            },
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(OpenAICompatibleJudge, "judge", fake_judge)
    config = JudgeConfig(
        base_url="http://localhost:1/v1", api_key="test", model_id="test", concurrency=1
    )
    output = tmp_path / "output"
    summary = judge_dataset(input_path=path, output_dir=output, config=config)
    assert summary["unresolved_task_count"] == 1
    assert summary["incomplete_query_count"] == 1
    assert not summary["training_dataset_ready"]
    assert (output / "ranking_dataset.llm_weak.jsonl").read_text() == ""
    assert "sensitive" not in (output / "failures.jsonl").read_text()
    fail.clear()
    calls.clear()
    summary = judge_dataset(input_path=path, output_dir=output, config=config)
    assert calls == ["q1:b"]
    assert summary["unresolved_task_count"] == 0
    assert summary["training_dataset_ready"]
    assert len((output / "ranking_dataset.llm_weak.jsonl").read_text().splitlines()) == 3


def test_summary_preserves_raw_hard_requirement_contradictions(tmp_path, monkeypatch):
    task = _task(6)
    task.update(
        {
            "task_id": "q:p",
            "query_id": "q",
            "query_group_id": "q",
            "context": {"query_id": "q"},
            "candidate": {"product_id": "p", "category": "cpu"},
        }
    )
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(task) + "\n", encoding="utf-8")

    def fake_judge(_self, _task):
        return (
            {
                "relevance_grade": 4,
                "confidence": 0.9,
                "explanation": "test",
                "missing_or_unsupported_information": [],
            },
            {"prompt_tokens": 1, "completion_tokens": 1},
        )

    monkeypatch.setattr(OpenAICompatibleJudge, "judge", fake_judge)
    config = JudgeConfig(base_url="http://localhost:1/v1", api_key="test", model_id="test")
    summary = judge_dataset(input_path=path, output_dir=tmp_path / "out", config=config)
    assert summary["raw_hard_requirement_contradiction_count"] == 1
    assert summary["raw_grade_recorded_count"] == 1
    assert summary["final_hard_requirement_contradiction_count"] == 0
    assert summary["grade_counts_by_category"]["cpu"]["0"] == 1


def test_circuit_breaker_stops_submitting_after_provider_failures(tmp_path, monkeypatch):
    tasks = []
    for index in range(20):
        task = _task(8)
        task.update(
            {
                "task_id": f"q:{index}",
                "query_id": "q",
                "query_group_id": "q",
                "context": {"query_id": "q"},
                "candidate": {"product_id": str(index), "category": "cpu"},
            }
        )
        tasks.append(task)
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")

    def fail(_self, _task):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(OpenAICompatibleJudge, "judge", fail)
    config = JudgeConfig(
        base_url="http://localhost:1/v1",
        api_key="test",
        model_id="test",
        concurrency=1,
        max_consecutive_failures=2,
    )
    summary = judge_dataset(input_path=path, output_dir=tmp_path / "out", config=config)
    assert summary["circuit_breaker_tripped"]
    assert summary["attempted_this_run"] == 2
    assert summary["not_attempted_this_run"] == 18
    assert summary["unresolved_task_count"] == 20


def test_schedule_balances_complete_groups_across_split_and_category(tmp_path):
    rows = [
        {"query_id": "a", "query_group_id": "a", "split": "train", "category": "cpu"},
        {"query_id": "b", "query_group_id": "b", "split": "train", "category": "cpu"},
        {"query_id": "c", "query_group_id": "c", "split": "test", "category": "gpu"},
    ]
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    tasks = [
        {
            "task_id": f"{r['query_id']}:{i}",
            "query_id": r["query_id"],
            "query_group_id": r["query_group_id"],
        }
        for r in rows
        for i in range(2)
    ]
    order = balance_pending_by_group(tasks, path)
    assert [x["query_id"] for x in order] == ["c", "c", "a", "a", "b", "b"]
