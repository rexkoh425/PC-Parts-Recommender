from __future__ import annotations

import json
import sys

from scripts.reuse_llm_judgments import main
from training.llm_judge import PROMPT_VERSION, RUBRIC, _canonical_hash


def test_reuses_only_identical_blinded_inputs(tmp_path, monkeypatch):
    old = []
    for pid in ("a", "b"):
        old.append(
            {
                "task_id": f"q:{pid}",
                "query_id": "q",
                "query_group_id": "q",
                "context": {"query_id": "q", "query_group_id": "q"},
                "candidate": {"product_id": pid, "retrieval_scores": {"bm25_score": 1}},
                "judge_input": {
                    "specifications": {},
                    "requirements": [],
                    "product": {"name": pid, "category": "cpu"},
                },
            }
        )
    new = json.loads(json.dumps(old))
    new[0]["candidate"]["retrieval_scores"]["bm25_score"] = 2
    new[1]["judge_input"]["product"]["name"] = "different product"

    def task_hash(task):
        return _canonical_hash(
            {
                "task": task,
                "model_id": "model",
                "prompt_version": PROMPT_VERSION,
                "rubric_sha256": _canonical_hash(RUBRIC),
            }
        )

    records = [
        {
            "task_id": t["task_id"],
            "task_hash": task_hash(t),
            "candidate": t["candidate"],
            "context": t["context"],
            "query_group_id": "q",
            "judgment": {"model_id": "model", "prompt_version": PROMPT_VERSION},
        }
        for t in old
    ]
    for name, rows in (("old.jsonl", old), ("new.jsonl", new), ("results.jsonl", records)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reuse",
            "--queue",
            str(tmp_path / "new.jsonl"),
            "--source",
            str(tmp_path / "results.jsonl"),
            "--source-queue",
            str(tmp_path / "old.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "model",
        ],
    )
    main()
    reused = [
        json.loads(x)
        for x in (tmp_path / "out/judgments.checkpoint.jsonl").read_text().splitlines()
    ]
    assert len(reused) == 1
    assert reused[0]["task_id"] == "q:a"
    assert reused[0]["task_hash"] == task_hash(new[0])
    assert reused[0]["candidate"]["retrieval_scores"]["bm25_score"] == 2
