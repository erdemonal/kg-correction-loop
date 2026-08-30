from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments" / "sosa_ssn_sampling_spec.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {value}") from exc
    return path


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def verify_input(item: dict) -> Path:
    path = repository_path(item["path"])
    if not path.is_file():
        raise RuntimeError(f"missing input: {item['path']}")
    actual = sha256_file(path)
    if actual != item["sha256"]:
        raise RuntimeError(
            f"input hash mismatch for {item['path']}: "
            f"expected {item['sha256']}, got {actual}"
        )
    return path


def unique_index(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    index = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{label}: missing {key}")
        if value in index:
            raise RuntimeError(f"{label}: duplicate {key} {value}")
        index[value] = row
    return index


def rank_key(seed: int, condition: str, scenario: str, source_unit_id: str) -> tuple:
    payload = f"{seed}|{condition}|{scenario}|{source_unit_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), source_unit_id


def candidate_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    index = {}
    for row in rows:
        key = (row.get("source_unit_id"), row.get("condition"))
        if not all(isinstance(value, str) and value for value in key):
            raise RuntimeError("candidate row missing source_unit_id or condition")
        if key in index:
            raise RuntimeError(f"duplicate source-condition candidate: {key}")
        if row.get("selection_status") != "eligible_not_selected":
            raise RuntimeError(f"candidate has unexpected status: {key}")
        index[key] = row
    return index


def selection_row(
    unit: dict,
    candidate: dict,
    condition: str,
    within_quota_rank: int | None,
    selection_basis: str,
) -> dict:
    record = unit.get("normalized_record", {})
    return {
        "case_id": f"sosa_ssn::{condition}::{unit['source_unit_id']}",
        "condition": condition,
        "source_unit_id": unit["source_unit_id"],
        "source_family": unit["source_family"],
        "scenario_family": unit["scenario_family"],
        "monitoring_location_id": record.get("monitoring_location_id"),
        "source_text_sha256": unit["source_text_sha256"],
        "raw_sha256": unit["raw_sha256"],
        "candidate_id": candidate["candidate_id"],
        "selection_basis": selection_basis,
        "within_quota_rank": within_quota_rank,
    }


def select(spec: dict, units: list[dict], candidates: list[dict]) -> list[dict]:
    unit_by_id = unique_index(units, "source_unit_id", "source units")
    candidates_by_pair = candidate_index(candidates)
    selected = []
    used = set()

    w3c_flat = [
        source_id
        for condition in spec["condition_order"]
        for source_id in spec["w3c_assignments"][condition]
    ]
    if len(w3c_flat) != len(set(w3c_flat)):
        raise RuntimeError("W3C source unit assigned to more than one condition")
    if len(w3c_flat) != spec["balance_targets"]["w3c_examples_total"]:
        raise RuntimeError("unexpected W3C assignment total")

    for condition in spec["condition_order"]:
        for source_id in spec["w3c_assignments"][condition]:
            unit = unit_by_id.get(source_id)
            if unit is None or unit.get("source_family") != "w3c_examples":
                raise RuntimeError(f"invalid W3C assignment: {condition}: {source_id}")
            candidate = candidates_by_pair.get((source_id, condition))
            if candidate is None:
                raise RuntimeError(f"ineligible W3C assignment: {condition}: {source_id}")
            if source_id in used:
                raise RuntimeError(f"source unit reused: {source_id}")
            used.add(source_id)
            selected.append(
                selection_row(
                    unit,
                    candidate,
                    condition,
                    within_quota_rank=None,
                    selection_basis="fixed_semantic_w3c_assignment",
                )
            )

    seed = spec["selection_seed"]
    for condition in spec["condition_order"]:
        for scenario, quota in spec["usgs_scenario_quotas"][condition].items():
            eligible = []
            for source_id, unit in unit_by_id.items():
                if source_id in used:
                    continue
                if unit.get("source_family") != "usgs_daily":
                    continue
                if unit.get("scenario_family") != scenario:
                    continue
                candidate = candidates_by_pair.get((source_id, condition))
                if candidate is None:
                    continue
                eligible.append((rank_key(seed, condition, scenario, source_id), unit, candidate))
            eligible.sort(key=lambda row: row[0])
            if len(eligible) < quota:
                raise RuntimeError(
                    f"insufficient unused candidates for {condition}/{scenario}: "
                    f"need {quota}, found {len(eligible)}"
                )
            for rank, (_key, unit, candidate) in enumerate(eligible[:quota], start=1):
                source_id = unit["source_unit_id"]
                if source_id in used:
                    raise RuntimeError(f"source unit reused: {source_id}")
                used.add(source_id)
                selected.append(
                    selection_row(
                        unit,
                        candidate,
                        condition,
                        within_quota_rank=rank,
                        selection_basis="seeded_sha256_within_fixed_usgs_quota",
                    )
                )

    return sorted(selected, key=lambda row: (row["condition"], row["source_unit_id"]))


def validate_selection(spec: dict, rows: list[dict]) -> None:
    expected_total = spec["sample_size"]["total_source_units"]
    expected_per_condition = spec["sample_size"]["per_condition"]
    if len(rows) != expected_total:
        raise RuntimeError(f"expected {expected_total} rows, found {len(rows)}")
    source_ids = [row["source_unit_id"] for row in rows]
    case_ids = [row["case_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("selected source unit reused across conditions")
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("duplicate selected case ID")

    by_condition = Counter(row["condition"] for row in rows)
    expected_conditions = set(spec["conditions"])
    if set(by_condition) != expected_conditions:
        raise RuntimeError("selected condition set changed")
    if any(by_condition[condition] != expected_per_condition for condition in expected_conditions):
        raise RuntimeError(f"condition allocation mismatch: {dict(by_condition)}")

    by_source = Counter(row["source_family"] for row in rows)
    if by_source != {
        "usgs_daily": spec["balance_targets"]["usgs_total"],
        "w3c_examples": spec["balance_targets"]["w3c_examples_total"],
    }:
        raise RuntimeError(f"source-family balance mismatch: {dict(by_source)}")

    usgs = [row for row in rows if row["source_family"] == "usgs_daily"]
    locations = [row["monitoring_location_id"] for row in usgs]
    if len(locations) != spec["balance_targets"]["distinct_monitoring_locations"]:
        raise RuntimeError("unexpected USGS monitoring-location count")
    if None in locations or len(locations) != len(set(locations)):
        raise RuntimeError("USGS monitoring locations are not unique")

    by_usgs_scenario = Counter(row["scenario_family"] for row in usgs)
    low, high = spec["balance_targets"]["usgs_scenario_total_range"]
    if min(by_usgs_scenario.values()) < low or max(by_usgs_scenario.values()) > high:
        raise RuntimeError(f"USGS scenario balance mismatch: {dict(by_usgs_scenario)}")


def counts(rows: list[dict], key: str) -> dict:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def nested_counts(rows: list[dict], first: str, second: str) -> dict:
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[row[first]][row[second]] += 1
    return {
        group: dict(sorted(values.items()))
        for group, values in sorted(grouped.items())
    }


def build(spec_path: Path = DEFAULT_SPEC) -> dict:
    spec = read_json(spec_path)
    if any(spec["execution"].values()):
        raise RuntimeError("sampling stage execution guard changed")
    if spec["source_unit_reuse"] is not False:
        raise RuntimeError("source unit reuse must remain disabled")
    if spec["sample_size"] != {
        "conditions": 6,
        "per_condition": 30,
        "total_source_units": 180,
        "one_case_equals_percentage_points": 3.3333333333333335,
        "maximum_wilson_95_half_width_at_n_30": 0.1685,
        "interpretation": "A controlled confirmatory characterization with finer descriptive resolution than the preliminary ten case cells. It is not a population representative power calculation.",
    }:
        raise RuntimeError("sample size block changed")

    input_paths = {name: verify_input(item) for name, item in spec["inputs"].items()}
    source_manifest = read_json(input_paths["source_manifest"])
    if source_manifest["selection"]["performed"] is not False:
        raise RuntimeError("input source pool already reports a selection")
    if source_manifest["source_units"]["count"] != 371:
        raise RuntimeError("source pool denominator changed")

    units = read_jsonl(input_paths["source_units"])
    candidates = read_jsonl(input_paths["candidate_pool"])
    rows = select(spec, units, candidates)
    validate_selection(spec, rows)

    selection_path = repository_path(spec["outputs"]["selection"])
    manifest_path = repository_path(spec["outputs"]["manifest"])
    write_jsonl(selection_path, rows)
    manifest = {
        "version": 1,
        "sampling_spec_sha256": sha256_file(spec_path),
        "source_pool_commit": spec["source_pool_commit"],
        "input_sha256": {name: item["sha256"] for name, item in sorted(spec["inputs"].items())},
        "selection": {
            "path": spec["outputs"]["selection"],
            "sha256": sha256_file(selection_path),
            "source_units": len(rows),
            "unique_source_units": len({row["source_unit_id"] for row in rows}),
            "by_condition": counts(rows, "condition"),
            "by_source_family": counts(rows, "source_family"),
            "by_scenario_family": counts(rows, "scenario_family"),
            "condition_by_source_family": nested_counts(rows, "condition", "source_family"),
            "condition_by_scenario_family": nested_counts(rows, "condition", "scenario_family"),
        },
        "sample_size_rationale": spec["sample_size"],
        "selection_method": {
            "seed": spec["selection_seed"],
            "description": spec["selection_method"],
            "uses_model_outcomes": False,
            "uses_human_annotations": False,
            "source_unit_reuse": False,
        },
        "preliminary_cases_included": 0,
        "execution": {
            "model_executed": False,
            "validator_executed": False,
            "reasoner_executed": False,
            "grounding_assessor_executed": False,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    manifest = build()
    selection = manifest["selection"]
    print(f"selected source units: {selection['source_units']}")
    print(f"unique source units: {selection['unique_source_units']}")
    print("by condition:")
    for condition, count in selection["by_condition"].items():
        print(f"  {condition}: {count}")
    print("by source family:")
    for family, count in selection["by_source_family"].items():
        print(f"  {family}: {count}")
    print("No model, validator, reasoner, or grounding assessor was executed.")


if __name__ == "__main__":
    main()
