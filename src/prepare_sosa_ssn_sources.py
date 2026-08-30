from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments" / "sosa_ssn_source_spec.json"
CONDITIONS = (
    "disjointness",
    "functional_property_conflict",
    "domain_range",
    "cardinality",
    "temporal",
    "grounding",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def repository_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {value}") from exc
    return path


def verify_file(path_value: str, expected_sha256: str) -> Path:
    path = repository_path(path_value)
    if not path.is_file():
        raise RuntimeError(f"missing frozen source: {path_value}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"frozen source hash mismatch: {path_value}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return path


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def clean_text(value) -> str:
    return " ".join(str(value).strip().split())


def render_usgs(record: dict, parameter: dict, metadata: dict) -> str:
    props = metadata["properties"]
    statistic = record["statistic_id"]
    statistic_phrase = {
        "00001": "maximum",
        "00002": "minimum",
        "00003": "mean",
        "00006": "sum",
        "00008": "median",
        "00011": "instantaneous",
    }.get(statistic, f"statistic {statistic}")
    return clean_text(
        f"On {record['time']}, the U.S. Geological Survey recorded an approved "
        f"daily {statistic_phrase} {parameter['name']} value of {record['value']} "
        f"{record['unit_of_measure']} at monitoring location "
        f"{record['monitoring_location_id']}. The observation belongs to time "
        f"series {record['time_series_id']} and uses parameter code "
        f"{record['parameter_code']} ({props['parameter_description']})."
    )


def normalize_usgs_feature(feature: dict, parameter_code: str) -> dict:
    if feature.get("type") != "Feature" or not isinstance(feature.get("properties"), dict):
        raise RuntimeError(f"invalid USGS feature in parameter {parameter_code}")
    props = feature["properties"]
    required = {
        "time_series_id",
        "monitoring_location_id",
        "parameter_code",
        "statistic_id",
        "time",
        "value",
        "unit_of_measure",
        "approval_status",
        "qualifier",
    }
    missing = sorted(required - set(props))
    if missing:
        raise RuntimeError(f"USGS feature missing {missing}")
    if props["parameter_code"] != parameter_code:
        raise RuntimeError(
            f"USGS parameter mismatch: expected {parameter_code}, "
            f"got {props['parameter_code']}"
        )
    record = {key: props[key] for key in sorted(required)}
    record["feature_id"] = feature.get("id")
    record["geometry"] = feature.get("geometry")
    return record


def usgs_units(spec: dict) -> tuple[list[dict], list[dict]]:
    family = spec["source_families"]["usgs_daily"]
    fixed_date = family["observation_date"]
    quality = family["quality_filter"]
    priority = {code: index for index, code in enumerate(family["parameter_priority"])}
    eligible_records = []
    rejected = []

    for parameter_code, parameter in sorted(family["parameters"].items()):
        snapshot_path = verify_file(parameter["snapshot"], parameter["snapshot_sha256"])
        metadata_path = verify_file(
            parameter["metadata_snapshot"], parameter["metadata_sha256"]
        )
        snapshot = read_json(snapshot_path)
        metadata = read_json(metadata_path)
        if snapshot.get("type") != "FeatureCollection":
            raise RuntimeError(f"not a FeatureCollection: {snapshot_path}")
        features = snapshot.get("features")
        if not isinstance(features, list) or len(features) != family["limit_per_parameter"]:
            raise RuntimeError(
                f"unexpected feature count for {parameter_code}: "
                f"{len(features) if isinstance(features, list) else 'not a list'}"
            )
        if metadata.get("properties", {}).get("id") != parameter_code:
            raise RuntimeError(f"parameter metadata mismatch: {metadata_path}")

        for feature in features:
            record = normalize_usgs_feature(feature, parameter_code)
            stable_key = f"{record['time_series_id']}:{record['time']}"
            if record["time"] != fixed_date:
                rejected.append(
                    {
                        "source_family": "usgs_daily",
                        "record_id": stable_key,
                        "reason": "observation_date_mismatch",
                    }
                )
                continue
            if record["approval_status"] != quality["approval_status"]:
                rejected.append(
                    {
                        "source_family": "usgs_daily",
                        "record_id": stable_key,
                        "reason": "not_approved",
                    }
                )
                continue
            if record["qualifier"] != quality["qualifier"]:
                rejected.append(
                    {
                        "source_family": "usgs_daily",
                        "record_id": stable_key,
                        "reason": "qualified_record",
                    }
                )
                continue
            if not all(
                clean_text(record[key])
                for key in (
                    "time_series_id",
                    "monitoring_location_id",
                    "statistic_id",
                    "value",
                    "unit_of_measure",
                )
            ):
                rejected.append(
                    {
                        "source_family": "usgs_daily",
                        "record_id": stable_key,
                        "reason": "missing_required_value",
                    }
                )
                continue
            eligible_records.append(
                {
                    "record": record,
                    "parameter": parameter,
                    "metadata": metadata,
                    "raw_path": parameter["snapshot"],
                    "raw_sha256": parameter["snapshot_sha256"],
                    "snapshot_timestamp": snapshot.get("timeStamp"),
                    "priority": priority[parameter_code],
                }
            )

    eligible_records.sort(
        key=lambda row: (
            row["record"]["monitoring_location_id"],
            row["priority"],
            row["record"]["time_series_id"],
        )
    )
    retained = []
    seen_sites = set()
    for row in eligible_records:
        record = row["record"]
        site = record["monitoring_location_id"]
        if site in seen_sites:
            rejected.append(
                {
                    "source_family": "usgs_daily",
                    "record_id": f"{record['time_series_id']}:{record['time']}",
                    "reason": "duplicate_monitoring_location",
                }
            )
            continue
        seen_sites.add(site)
        retained.append(row)

    units = []
    for row in retained:
        record = row["record"]
        parameter = row["parameter"]
        source_text = render_usgs(record, parameter, row["metadata"])
        units.append(
            {
                "source_unit_id": f"usgs:{record['time_series_id']}:{record['time']}",
                "source_family": "usgs_daily",
                "scenario_family": parameter["scenario_family"],
                "public_source_identifier": (
                    f"USGS daily:{record['time_series_id']}:{record['time']}"
                ),
                "upstream_query_url": parameter["query_url"],
                "raw_path": row["raw_path"],
                "raw_sha256": row["raw_sha256"],
                "snapshot_captured_at_utc": row["snapshot_timestamp"],
                "normalized_record": record,
                "source_text": source_text,
                "source_text_sha256": sha256_bytes(source_text.encode("utf-8")),
                "eligible_conditions": list(family["eligible_conditions"]),
                "eligibility_basis": "record derived observation and one day observation collection wrapper",
            }
        )
    return units, sorted(rejected, key=lambda row: (row["reason"], row["record_id"]))


def w3c_units(spec: dict) -> list[dict]:
    family = spec["source_families"]["w3c_examples"]
    units = []
    for example in family["examples"]:
        path = verify_file(example["path"], example["sha256"])
        raw = path.read_text(encoding="utf-8")
        if example["root_token"] not in raw:
            raise RuntimeError(
                f"W3C root token {example['root_token']} absent from {example['path']}"
            )
        conditions = example["eligible_conditions"]
        unknown = sorted(set(conditions) - set(CONDITIONS))
        if unknown:
            raise RuntimeError(f"unknown W3C conditions for {example['id']}: {unknown}")
        source_text = clean_text(example["summary"])
        units.append(
            {
                "source_unit_id": example["id"],
                "source_family": "w3c_examples",
                "scenario_family": example["scenario_family"],
                "public_source_identifier": (
                    f"{family['repository']}/blob/{family['pinned_commit']}/"
                    f"ssn/rdf/examples/{Path(example['path']).name}#{example['root_token']}"
                ),
                "upstream_query_url": None,
                "raw_path": example["path"],
                "raw_sha256": example["sha256"],
                "snapshot_commit": family["pinned_commit"],
                "normalized_record": {
                    "root_token": example["root_token"],
                    "summary": source_text,
                },
                "source_text": source_text,
                "source_text_sha256": sha256_bytes(source_text.encode("utf-8")),
                "eligible_conditions": list(conditions),
                "eligibility_basis": "curated root in a byte-pinned official W3C example",
            }
        )
    return units


def candidate_rows(units: list[dict]) -> list[dict]:
    rows = []
    for unit in units:
        for condition in unit["eligible_conditions"]:
            rows.append(
                {
                    "candidate_id": f"{unit['source_unit_id']}::{condition}",
                    "source_unit_id": unit["source_unit_id"],
                    "source_family": unit["source_family"],
                    "scenario_family": unit["scenario_family"],
                    "condition": condition,
                    "source_text_sha256": unit["source_text_sha256"],
                    "eligibility_basis": unit["eligibility_basis"],
                    "selection_status": "eligible_not_selected",
                }
            )
    return sorted(rows, key=lambda row: (row["condition"], row["source_unit_id"]))


def counts(rows: list[dict], key: str) -> dict:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def validate_units_and_candidates(spec: dict, units: list[dict], candidates: list[dict]) -> None:
    unit_ids = [unit["source_unit_id"] for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise RuntimeError("duplicate source_unit_id")
    candidate_ids = [row["candidate_id"] for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("duplicate candidate_id")
    known_units = set(unit_ids)
    for row in candidates:
        if row["source_unit_id"] not in known_units:
            raise RuntimeError("candidate refers to unknown source unit")
        if row["condition"] not in CONDITIONS:
            raise RuntimeError("candidate has unknown condition")
    condition_counts = Counter(row["condition"] for row in candidates)
    target = spec["sampling_constraints_not_yet_applied"]
    if not target:
        raise RuntimeError("missing sampling constraints")
    under_target = {condition: condition_counts[condition] for condition in CONDITIONS if condition_counts[condition] < 30}
    if under_target:
        raise RuntimeError(f"candidate pool below design target: {under_target}")


def build(spec_path: Path = DEFAULT_SPEC) -> dict:
    spec = read_json(spec_path)
    if spec["execution"] != {
        "offline_only": True,
        "runs_extractor": False,
        "runs_repair_model": False,
        "runs_grounding_assessor": False,
        "runs_validator": False,
        "selects_final_cases": False,
        "locks_sample_size": False,
    }:
        raise RuntimeError("source-preparation execution guard changed")
    if tuple(spec["conditions"]) != CONDITIONS:
        raise RuntimeError("condition list changed")

    usgs, rejected = usgs_units(spec)
    w3c = w3c_units(spec)
    units = sorted(usgs + w3c, key=lambda row: row["source_unit_id"])
    candidates = candidate_rows(units)
    validate_units_and_candidates(spec, units, candidates)

    outputs = spec["outputs"]
    units_path = repository_path(outputs["source_units"])
    candidates_path = repository_path(outputs["candidate_pool"])
    manifest_path = repository_path(outputs["manifest"])
    write_jsonl(units_path, units)
    write_jsonl(candidates_path, candidates)

    manifest = {
        "version": 1,
        "spec_sha256": sha256_file(spec_path),
        "ontology_commit": spec["ontology_commit"],
        "adapter_version": spec["adapter_version"],
        "renderer_version": spec["renderer_version"],
        "experimental_unit": "source_unit",
        "source_units": {
            "count": len(units),
            "by_source_family": counts(units, "source_family"),
            "by_scenario_family": counts(units, "scenario_family"),
            "path": outputs["source_units"],
            "sha256": sha256_file(units_path),
        },
        "candidate_pool": {
            "count": len(candidates),
            "by_condition": counts(candidates, "condition"),
            "by_source_family": counts(candidates, "source_family"),
            "path": outputs["candidate_pool"],
            "sha256": sha256_file(candidates_path),
            "rows_are_independent_experimental_units": False,
        },
        "usgs_rejections": {
            "count": len(rejected),
            "by_reason": counts(rejected, "reason"),
            "records": rejected,
        },
        "source_family_status": {
            "usgs_daily": "included",
            "w3c_examples": "included",
            "epa_airdata": "deferred_not_counted",
        },
        "selection": {
            "performed": False,
            "sample_size_locked": False,
            "one_condition_per_source_unit_required_later": True,
        },
        "execution": {
            "network_accessed": False,
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
    print(f"source units: {manifest['source_units']['count']}")
    print(f"candidate rows: {manifest['candidate_pool']['count']}")
    print("by condition:")
    for condition, count in manifest["candidate_pool"]["by_condition"].items():
        print(f"  {condition}: {count}")
    print(f"rejected USGS records: {manifest['usgs_rejections']['count']}")
    print("No model, validator, reasoner, or grounding assessor was executed.")


if __name__ == "__main__":
    main()
