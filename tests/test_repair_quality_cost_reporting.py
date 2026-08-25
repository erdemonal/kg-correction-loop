import json
from pathlib import Path

from src.analyze_repair_quality_cost import PRIMARY_CONDITIONS
from src.report_repair_quality_cost import (
    EXPECTED_ARTIFACTS,
    empty_cases,
    empty_reference_rows,
    draft_results_notes,
    join_dynamics,
    load_spec,
    nonempty_cases,
    quality_cost_rows,
    run_report,
    validate_analysis,
)


def test_spec_keeps_plain_view_and_forbids_reruns():
    spec = load_spec()
    assert spec["view"] == "clean reference F1 and repair side effects"
    assert spec["execution"]["runs_repair_model"] is False
    assert spec["execution"]["runs_validator"] is False
    rules = " ".join(spec["reporting_rules"])
    assert "human source faithfulness" in rules
    assert "Text2KGBench F1" in rules
    assert "does not define a Pareto front" in rules
    assert "not population confidence intervals" in rules
    assert "not wall clock time" in rules
    assert "do not appear in the main F1 figures" in rules


def test_nonempty_cases_exclude_empty_references():
    cases = [
        {"id": "a", "empty_reference": False, "initial_reference_size": 2, "condition": "temporal"},
        {"id": "b", "empty_reference": True, "initial_reference_size": 0, "condition": "domain_range"},
    ]
    selected = nonempty_cases(cases)
    assert [row["id"] for row in selected] == ["a"]
    vacant = empty_cases(cases)
    assert [row["id"] for row in vacant] == ["b"]


def test_frozen_primary_tables_match_verified_headlines():
    payload = json.loads(Path("results/repair_quality_cost.json").read_text(encoding="utf-8"))
    dynamics = json.loads(Path("results/repair_dynamics_analysis.json").read_text(encoding="utf-8"))
    validate_analysis(payload)
    cases = join_dynamics(payload, dynamics)
    primary = nonempty_cases(cases)
    assert len(primary) == 40
    assert all(row["initial_reference_size"] > 0 for row in primary)
    assert all(row["condition"] != "domain_range" for row in primary)
    assert "domain_range" not in payload["primary_f1"]["by_condition"]
    rows = quality_cost_rows(payload, cases)
    overall = rows[0]
    assert overall["n"] == 40
    assert overall["improved"] == 25
    assert overall["unchanged"] == 6
    assert overall["worsened"] == 9
    assert round(overall["mean_initial_f1"], 3) == 0.590
    assert round(overall["mean_last_validated_f1"], 3) == 0.833
    assert round(overall["mean_f1_delta"], 3) == 0.244
    assert overall["bootstrap_samples"] == 10000
    assert overall["bootstrap_seed"] == 42
    assert [row["condition"] for row in rows[1:]] == list(PRIMARY_CONDITIONS)
    empty = empty_reference_rows(payload)[0]
    assert empty["n"] == 10
    assert empty["primary_metric"] == "extra triples, not F1"
    assert empty["exact_empty_graph_recovery"] == 0
    assert empty["mean_initial_extra_triples"] == 1.0
    assert empty["mean_last_validated_extra_triples"] == 3.3
    assert empty["improved"] == 0
    assert empty["unchanged"] == 2
    assert empty["worsened"] == 8
    assert empty["output_failure"] == 5


def test_notes_carry_required_warnings():
    payload = json.loads(Path("results/repair_quality_cost.json").read_text(encoding="utf-8"))
    dynamics = json.loads(Path("results/repair_dynamics_analysis.json").read_text(encoding="utf-8"))
    text = draft_results_notes(payload, join_dynamics(payload, dynamics))
    assert "how close the repaired graph is to the controlled clean graph" in text
    assert "not a human judgment of source faithfulness" in text
    assert "not Text2KGBench F1" in text
    assert "does not define a Pareto front" in text
    assert "not a population confidence interval" in text
    assert "not wall clock time" in text
    assert "do not appear in the main F1 figures" in text
    assert "extra triple counts" in text
    assert "not a follow up of the same 40 cases" in text
    assert "0.590" in text
    assert "0.833" in text


def test_report_writes_expected_artifacts(tmp_path):
    manifest = run_report(output_dir=tmp_path)
    names = set(manifest["outputs"])
    assert names == set(EXPECTED_ARTIFACTS)
    for name in EXPECTED_ARTIFACTS:
        assert (tmp_path / name).exists()
    notes = (tmp_path / "repair_quality_cost_notes.md").read_text(encoding="utf-8")
    assert "Pareto front" in notes
    assert "empty clean reference" in notes or "domain_range" in notes
    csv_text = (tmp_path / "quality_cost_summary.csv").read_text(encoding="utf-8")
    assert "domain_range" not in csv_text
    empty_csv = (tmp_path / "empty_reference_summary.csv").read_text(encoding="utf-8")
    assert "extra triples, not F1" in empty_csv
    assert manifest["primary_n"] == 40
    assert manifest["empty_reference_n"] == 10
    assert manifest["models_or_validators_run"] is False
