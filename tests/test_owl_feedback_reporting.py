import copy

import pytest

from src.report_owl_feedback_framing import (
    CONDITIONS,
    condition_rows,
    domain_rows,
    draft_results_notes,
    paired_rows,
    short_case_label,
    validate_analysis,
)


def make_summary(target, usable, owl, reference, collateral, shacl, grounding, failures, n=10):
    counts = {
        "n": n,
        "usable_outputs": usable,
        "output_failures": failures,
        "controlled_target_removed": target,
        "reference_recovery": reference,
        "owl_consistent": owl,
        "collateral_edit": collateral,
        "new_raw_shacl_findings": shacl,
        "new_grounding_findings": grounding,
        "owl_inconsistent_after_target_removal": 2 if target >= 2 else 0,
        "output_failure_types": {},
    }
    rates = {
        key: {"numerator": counts[key], "denominator": usable, "estimate": counts[key] / usable}
        for key in ("owl_consistent", "collateral_edit", "new_raw_shacl_findings", "new_grounding_findings")
    }
    return {"counts": counts, "rates": rates, "edits": {"mean_from_injected": 4.5}}


def make_payload():
    by_condition = {
        "verdict": make_summary(8, 8, 6, 0, 8, 3, 8, 2),
        "location": make_summary(8, 9, 6, 0, 8, 2, 8, 1),
        "explanation": make_summary(9, 9, 7, 1, 8, 1, 8, 1),
    }
    comparisons = {
        "explanation_vs_verdict": ("explanation", "verdict", 8, 1, 0, 1),
        "explanation_vs_location": ("explanation", "location", 7, 2, 1, 0),
        "location_vs_verdict": ("location", "verdict", 7, 1, 1, 1),
    }
    paired = {
        name: {
            "left": left,
            "right": right,
            "n_paired_cases": 10,
            "both_resolved": both,
            "left_only": left_only,
            "right_only": right_only,
            "neither_resolved": neither,
            "same": both + neither,
            "net_target_difference": left_only - right_only,
        }
        for name, (left, right, both, left_only, right_only, neither) in comparisons.items()
    }
    domains = {
        domain: {condition: make_summary(4, 5, 3, 0, 4, 1, 4, 0, n=5) for condition in CONDITIONS}
        for domain in ("movie", "music")
    }
    return {
        "analysis_unit": "controlled case",
        "paired_by_case": True,
        "n_paired_cases": 10,
        "case_condition_observations": 30,
        "conditions": list(CONDITIONS),
        "by_condition": by_condition,
        "by_domain": domains,
        "paired_target_comparisons": paired,
        "pooled_case_condition_observations": {
            "counts": {"output_failure_types": {"unparseable_output": 3, "relation_outside_allowed_set": 1}}
        },
        "residual_owl_case_ids": ["ont_2_music_test_125", "ont_2_music_test_603"],
        "cases": [{"id": f"case-{index}"} for index in range(10)],
    }


def test_validate_analysis_accepts_complete_paired_payload():
    validate_analysis(make_payload())


def test_validate_analysis_rejects_graph_denominator_of_all_cases():
    payload = make_payload()
    payload["by_condition"]["verdict"]["rates"]["owl_consistent"]["denominator"] = 10
    with pytest.raises(RuntimeError, match="Denominator for outcomes that depend on a parsed graph"):
        validate_analysis(payload)


def test_condition_table_keeps_usable_and_all_case_denominators_separate():
    rows = {row["condition"]: row for row in condition_rows(make_payload())}
    assert rows["verdict"]["target_removed"] == "8/10"
    assert rows["verdict"]["owl_consistent_given_usable"] == "6/8"
    assert rows["verdict"]["collateral_given_usable"] == "8/8"
    assert rows["explanation"]["exact_reference_recovery"] == "1/10"


def test_paired_table_preserves_both_directions_of_discordance():
    rows = {row["comparison"]: row for row in paired_rows(make_payload())}
    explanation = rows["explanation_vs_location"]
    assert (explanation["left_only"], explanation["right_only"]) == (2, 1)
    assert explanation["paired_cases"] == 10


def test_domain_table_has_six_domain_condition_rows():
    rows = domain_rows(make_payload())
    assert len(rows) == 6
    assert {row["domain"] for row in rows} == {"movie", "music"}


def test_results_notes_keep_claim_boundaries():
    notes = draft_results_notes(make_payload())
    assert "6/8" in notes
    assert "7/9" in notes
    assert "the same 2 paired Music cases" in notes
    assert "not 30 independent experimental units" in notes
    assert "not a newly generated reasoner explanation" in notes
    assert "human review of each assertion" in notes
    assert "statistical dominance" in notes
    assert "exact source graph" not in notes


def test_short_case_labels_preserve_domain_and_identifier():
    assert short_case_label("ont_2_music_test_603") == "Music 603"


def test_failure_count_must_balance_usable_outputs():
    payload = copy.deepcopy(make_payload())
    payload["by_condition"]["explanation"]["counts"]["output_failures"] = 0
    with pytest.raises(RuntimeError, match="Usable outputs and failures"):
        validate_analysis(payload)
