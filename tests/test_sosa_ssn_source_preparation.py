import hashlib
import json
from collections import Counter
from pathlib import Path

from src.prepare_sosa_ssn_sources import (
    CONDITIONS,
    candidate_rows,
    read_json,
    usgs_units,
    validate_units_and_candidates,
    w3c_units,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "experiments" / "sosa_ssn_source_spec.json"
MANIFEST_PATH = ROOT / "experiments" / "sosa_ssn_source_manifest.json"
INVENTORY_PATH = ROOT / "experiments" / "sosa_ssn_axiom_inventory.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec():
    return read_json(SPEC_PATH)


def prepared():
    payload = spec()
    usgs, rejected = usgs_units(payload)
    w3c = w3c_units(payload)
    units = sorted(usgs + w3c, key=lambda row: row["source_unit_id"])
    candidates = candidate_rows(units)
    return payload, units, candidates, rejected


def test_source_stage_is_offline_and_does_not_select_or_execute():
    payload = spec()
    execution = payload["execution"]

    assert execution["offline_only"] is True
    assert execution["selects_final_cases"] is False
    assert execution["locks_sample_size"] is False
    assert execution["runs_extractor"] is False
    assert execution["runs_repair_model"] is False
    assert execution["runs_grounding_assessor"] is False
    assert execution["runs_validator"] is False


def test_vendored_core_modules_match_the_pinned_inventory():
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    root = ROOT / inventory["vendored_core_root"]

    assert inventory["source_commit"] == "37fa55298187464b41c3712620dcbf5bd438b1b2"
    assert len(inventory["core_module_sha256"]) == 12
    for source_path, expected in inventory["core_module_sha256"].items():
        local = root / Path(source_path).name
        assert local.is_file()
        assert sha256(local) == expected


def test_frozen_usgs_snapshots_yield_359_unique_approved_unqualified_sites():
    _payload, units, _candidates, rejected = prepared()
    usgs = [row for row in units if row["source_family"] == "usgs_daily"]

    assert len(usgs) == 359
    sites = [row["normalized_record"]["monitoring_location_id"] for row in usgs]
    assert len(sites) == len(set(sites))
    assert all(row["normalized_record"]["approval_status"] == "Approved" for row in usgs)
    assert all(row["normalized_record"]["qualifier"] is None for row in usgs)
    assert Counter(row["scenario_family"] for row in usgs) == {
        "water_temperature": 95,
        "precipitation": 90,
        "stream_discharge": 82,
        "gage_height": 92,
    }
    assert Counter(row["reason"] for row in rejected) == {
        "not_approved": 17,
        "qualified_record": 16,
        "duplicate_monitoring_location": 8,
    }


def test_usgs_identifier_does_not_depend_on_the_unstable_feature_uuid():
    _payload, units, _candidates, _rejected = prepared()
    row = next(unit for unit in units if unit["source_family"] == "usgs_daily")
    record = row["normalized_record"]

    assert record["feature_id"] not in row["source_unit_id"]
    assert record["time_series_id"] in row["source_unit_id"]
    assert record["time"] in row["source_unit_id"]
    assert record["monitoring_location_id"] in row["source_text"]
    assert record["value"] in row["source_text"]


def test_twelve_pinned_w3c_examples_are_distinct_adapter_units():
    _payload, units, _candidates, _rejected = prepared()
    w3c = [row for row in units if row["source_family"] == "w3c_examples"]

    assert len(w3c) == 12
    assert len({row["source_unit_id"] for row in w3c}) == 12
    assert len({row["scenario_family"] for row in w3c}) == 12
    assert all(row["source_text"] for row in w3c)
    assert all("37fa55298187464b41c3712620dcbf5bd438b1b2" in row["public_source_identifier"] for row in w3c)


def test_candidate_rows_are_eligibilities_not_independent_outcomes():
    payload, units, candidates, _rejected = prepared()
    validate_units_and_candidates(payload, units, candidates)

    assert len(units) == 371
    assert len(candidates) == 2205
    assert Counter(row["condition"] for row in candidates) == {
        "cardinality": 371,
        "disjointness": 362,
        "domain_range": 371,
        "functional_property_conflict": 369,
        "grounding": 371,
        "temporal": 361,
    }
    assert all(row["selection_status"] == "eligible_not_selected" for row in candidates)
    assert all(row["condition"] in CONDITIONS for row in candidates)
    assert len({row["candidate_id"] for row in candidates}) == len(candidates)


def test_manifest_preserves_denominators_and_no_execution_claims():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["source_units"]["count"] == 371
    assert manifest["source_units"]["by_source_family"] == {
        "usgs_daily": 359,
        "w3c_examples": 12,
    }
    assert manifest["candidate_pool"]["count"] == 2205
    assert manifest["candidate_pool"]["rows_are_independent_experimental_units"] is False
    assert manifest["selection"]["performed"] is False
    assert manifest["selection"]["sample_size_locked"] is False
    assert not any(manifest["execution"].values())


def test_epa_is_explicitly_deferred_and_not_counted_as_an_implemented_family():
    payload = spec()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert payload["source_families"]["epa_airdata"]["status"] == "deferred_from_v1_pool"
    assert manifest["source_family_status"]["epa_airdata"] == "deferred_not_counted"
    assert "epa_airdata" not in manifest["source_units"]["by_source_family"]
