import json
from collections import Counter
from pathlib import Path

from src.select_sosa_ssn_confirmatory_cases import (
    read_json,
    read_jsonl,
    select,
    validate_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "experiments" / "sosa_ssn_sampling_spec.json"
SELECTION_PATH = ROOT / "experiments" / "sosa_ssn_confirmatory_selection.jsonl"
MANIFEST_PATH = ROOT / "experiments" / "sosa_ssn_sampling_manifest.json"


def inputs():
    spec = read_json(SPEC_PATH)
    units = read_jsonl(ROOT / spec["inputs"]["source_units"]["path"])
    candidates = read_jsonl(ROOT / spec["inputs"]["candidate_pool"]["path"])
    return spec, units, candidates


def test_sample_size_is_locked_to_six_cells_of_thirty():
    spec, _units, _candidates = inputs()

    assert spec["sample_size"]["conditions"] == 6
    assert spec["sample_size"]["per_condition"] == 30
    assert spec["sample_size"]["total_source_units"] == 180
    assert spec["sample_size"]["one_case_equals_percentage_points"] == 100 / 30
    assert "not a population representative power calculation" in spec["sample_size"]["interpretation"]


def test_selection_has_180_unique_source_units_and_thirty_per_condition():
    spec, _units, _candidates = inputs()
    rows = read_jsonl(SELECTION_PATH)
    validate_selection(spec, rows)

    assert len(rows) == 180
    assert len({row["source_unit_id"] for row in rows}) == 180
    assert Counter(row["condition"] for row in rows) == {
        condition: 30 for condition in spec["conditions"]
    }


def test_selection_uses_168_usgs_and_all_twelve_w3c_examples():
    _spec, units, _candidates = inputs()
    rows = read_jsonl(SELECTION_PATH)

    assert Counter(row["source_family"] for row in rows) == {
        "usgs_daily": 168,
        "w3c_examples": 12,
    }
    all_w3c = {
        unit["source_unit_id"]
        for unit in units
        if unit["source_family"] == "w3c_examples"
    }
    selected_w3c = {
        row["source_unit_id"]
        for row in rows
        if row["source_family"] == "w3c_examples"
    }
    assert selected_w3c == all_w3c


def test_usgs_scenarios_and_monitoring_locations_are_balanced_and_unique():
    _spec, _units, _candidates = inputs()
    rows = read_jsonl(SELECTION_PATH)
    usgs = [row for row in rows if row["source_family"] == "usgs_daily"]

    assert Counter(row["scenario_family"] for row in usgs) == {
        "water_temperature": 43,
        "precipitation": 42,
        "stream_discharge": 42,
        "gage_height": 41,
    }
    locations = [row["monitoring_location_id"] for row in usgs]
    assert None not in locations
    assert len(locations) == len(set(locations)) == 168


def test_every_selected_pair_was_eligible_in_the_frozen_candidate_pool():
    _spec, _units, candidates = inputs()
    rows = read_jsonl(SELECTION_PATH)
    eligible = {
        (row["source_unit_id"], row["condition"]): row["candidate_id"]
        for row in candidates
    }

    for row in rows:
        assert eligible[(row["source_unit_id"], row["condition"])] == row["candidate_id"]


def test_seeded_selection_is_independent_of_input_row_order():
    spec, units, candidates = inputs()

    forward = select(spec, units, candidates)
    reversed_inputs = select(spec, list(reversed(units)), list(reversed(candidates)))

    assert forward == reversed_inputs
    assert forward == read_jsonl(SELECTION_PATH)


def test_semantic_w3c_assignments_are_fixed_and_not_randomized():
    spec, _units, _candidates = inputs()

    assert spec["w3c_assignments"]["disjointness"] == [
        "w3c_planned_ph_collection"
    ]
    assert spec["w3c_assignments"]["temporal"] == [
        "w3c_eautonome_collection",
        "w3c_time_series_collection",
    ]
    assigned = [
        source_id
        for condition in spec["condition_order"]
        for source_id in spec["w3c_assignments"][condition]
    ]
    assert len(assigned) == len(set(assigned)) == 12


def test_manifest_records_no_experimental_execution_or_preliminary_cases():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["selection"]["sha256"] == (
        "99534641c555b4359bfc9dcf555ff93ba191e7bfbd39265394024507b11a4299"
    )
    assert manifest["preliminary_cases_included"] == 0
    assert manifest["selection_method"]["uses_model_outcomes"] is False
    assert manifest["selection_method"]["uses_human_annotations"] is False
    assert manifest["selection_method"]["source_unit_reuse"] is False
    assert not any(manifest["execution"].values())
