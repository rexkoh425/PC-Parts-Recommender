from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

from pipelines.sources.zenodo_er_benchmark import (
    EntityMatchingSourceDataset,
    SourcePairLabel,
    SourceProductRecord,
    _parse_archive,
    _stable_split,
    adapt_pair,
    record_disjoint_split,
)

SEED = 20260722


def _csv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode()


def _tiny_archive(path: Path) -> None:
    table_header = ["id", "title", "modelno", "price", "shipweight", "brand", "dimensions"]
    pair_header = ["id", "left_id", "right_id", "label"]
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            "Dn7/tableA.csv",
            _csv_bytes(
                table_header,
                [
                    ["0", '"Quoted Device A"', '"MPN-A"', "10.50", "1.2", '"Acme"', "1 x 2"],
                    ["1", "Device B", "MPN-B", "", "", "Acme", ""],
                    ["2", "Device C", "MPN-C", "12", "2", "Beta", ""],
                ],
            ),
        )
        archive.writestr(
            "Dn7/tableB.csv",
            _csv_bytes(
                table_header,
                [
                    ["0", "Device A", "MPN-A", "11", "1 206", "Acme", "1 x 2"],
                    ["1", "Different B", "MPN-X", "20", "2", "Other", ""],
                    ["2", "Device C", "MPN-C", "12", "2", "Beta", ""],
                ],
            ),
        )
        archive.writestr("Dn7/train_set.csv", _csv_bytes(pair_header, [["0", "0", "0", "1"]]))
        archive.writestr("Dn7/valid_set.csv", _csv_bytes(pair_header, [["0", "1", "1", "0"]]))
        archive.writestr("Dn7/test_set.csv", _csv_bytes(pair_header, [["0", "2", "2", "1"]]))


def _record(record_id: str) -> SourceProductRecord:
    return SourceProductRecord(
        record_id=record_id,
        title=f"Product {record_id}",
        model_number=f"MPN-{record_id}",
        price_usd=10.0,
        shipping_weight_lb=None,
        brand="Acme",
        dimensions=None,
    )


def _id_for_split(prefix: str, split_name: str, *, offset: int) -> str:
    for index in range(offset, offset + 10_000):
        record_id = f"{prefix}-{index}"
        if _stable_split(f"{prefix}:{record_id}", seed=SEED) == split_name:
            return record_id
    raise AssertionError(f"could not find deterministic {split_name} id")


def _record_disjoint_fixture() -> EntityMatchingSourceDataset:
    left: dict[str, SourceProductRecord] = {}
    right: dict[str, SourceProductRecord] = {}
    pairs: list[SourcePairLabel] = []
    for index, split_name in enumerate(("train", "validation", "test")):
        # A positive component's stable root is its lexicographically smaller left node.
        positive_left = _id_for_split("left", split_name, offset=index * 1_000)
        positive_right = f"positive-right-{index}"
        negative_left = _id_for_split("left", split_name, offset=4_000 + index * 1_000)
        negative_right = _id_for_split("right", split_name, offset=8_000 + index * 1_000)
        for record_id in (positive_left, negative_left):
            left[record_id] = _record(record_id)
        for record_id in (positive_right, negative_right):
            right[record_id] = _record(record_id)
        pairs.extend(
            (
                SourcePairLabel(
                    source_pair_id=f"positive-{index}",
                    left_id=positive_left,
                    right_id=positive_right,
                    label=1,
                    original_split="train",
                ),
                SourcePairLabel(
                    source_pair_id=f"negative-{index}",
                    left_id=negative_left,
                    right_id=negative_right,
                    label=0,
                    original_split="train",
                ),
            )
        )

    cross_left = _id_for_split("left", "train", offset=20_000)
    cross_right = _id_for_split("right", "test", offset=20_000)
    left[cross_left] = _record(cross_left)
    right[cross_right] = _record(cross_right)
    pairs.append(
        SourcePairLabel(
            source_pair_id="cross-negative",
            left_id=cross_left,
            right_id=cross_right,
            label=0,
            original_split="test",
        )
    )
    return EntityMatchingSourceDataset(left=left, right=right, pairs=tuple(pairs))


def test_tiny_archive_parser_preserves_labels_and_cleans_literal_quotes(tmp_path: Path) -> None:
    archive = tmp_path / "tiny-dn7.zip"
    _tiny_archive(archive)

    parsed = _parse_archive(archive)

    assert len(parsed.left) == 3
    assert len(parsed.right) == 3
    assert len(parsed.pairs) == 3
    assert parsed.left["0"].title == "Quoted Device A"
    assert parsed.left["0"].model_number == "MPN-A"
    assert parsed.left["0"].price_usd == 10.5
    assert parsed.left["0"].price_source_text == "10.50"
    assert parsed.right["0"].shipping_weight_lb is None
    assert parsed.right["0"].shipping_weight_source_text == "1 206"
    assert [pair.label for pair in parsed.pairs] == [1, 0, 1]
    assert [pair.original_split for pair in parsed.pairs] == ["train", "validation", "test"]
    adapted = adapt_pair(parsed.pairs[0], dataset=parsed, assigned_split="train")
    assert adapted.product.attributes["source_shipping_weight_lb"] is None
    assert adapted.product.attributes["source_shipping_weight_text"] == "1 206"


def test_record_disjoint_split_is_deterministic_and_drops_no_positive() -> None:
    dataset = _record_disjoint_fixture()

    first = record_disjoint_split(dataset, seed=SEED)
    second = record_disjoint_split(dataset, seed=SEED)

    assert first == second
    assert first.dropped_cross_split_pairs == 1
    assert first.dropped_cross_split_positives == 0
    assert sum(pair.label for rows in first.pairs.values() for pair in rows) == 3
    for split_name, rows in first.pairs.items():
        assert {pair.label for pair in rows} == {0, 1}, split_name

    for left_id, assigned in first.left_assignments.items():
        observed = {
            split_name
            for split_name, rows in first.pairs.items()
            if any(row.left_id == left_id for row in rows)
        }
        assert not observed or observed == {assigned}
    for right_id, assigned in first.right_assignments.items():
        observed = {
            split_name
            for split_name, rows in first.pairs.items()
            if any(row.right_id == right_id for row in rows)
        }
        assert not observed or observed == {assigned}


def test_adaptation_keeps_usd_out_of_sgd_fields() -> None:
    dataset = _record_disjoint_fixture()
    source_pair = dataset.pairs[0]

    adapted = adapt_pair(source_pair, dataset=dataset, assigned_split="train")

    assert adapted.listing.current_price_sgd is None
    assert adapted.product.price_sgd is None
    assert adapted.listing.attributes["source_currency"] == "USD"
    assert adapted.product.attributes["source_currency"] == "USD"
    assert adapted.listing.attributes["source_price_usd"] == 10.0
    assert not adapted.is_synthetic
