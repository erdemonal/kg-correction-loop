import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def test_protocol_records_main_limits():
    text = (EXPERIMENTS / "validation_protocol.md").read_text(
        encoding="utf-8"
    )

    required = [
        "The controlled validation setup was fixed before the repair experiments began.",
        "44 of the 50 controlled modifications",
        "29 of the 50 clean graphs",
        "21 of the 50 clean graphs",
        "19 of the 50 clean cases",
        "case level labels",
        "did not measure the assessor's accuracy separately for each triple",
        "Movie `narrative_location` cases with Country objects",
        "All five Music `record_label` cases behaved as expected",
        "does not examine the date assertions or their order",
        "pySHACL OWL RL inference enabled",
        "This review only explains the observed errors",
        "does not mean that their errors are statistically independent",
        "remain unchanged for two consecutive repair rounds",
        "experiments/environment.json",
        "experiments/validation_manifest.json",
    ]

    for phrase in required:
        assert phrase in text


def test_enrichment_spec_uses_plain_final_wording():
    text = (EXPERIMENTS / "enrichment_spec.md").read_text(
        encoding="utf-8"
    )

    assert "after OWL RL materialization" not in text
    assert "built-in OWL-RL" not in text
    assert "grounding-detectable" not in text
    assert "background diagnostics" not in text
    assert "pySHACL OWL RL inference enabled" in text
    assert (
        "The grounding assessor cannot detect the controlled deletion itself"
        in text
    )
    assert (
        "The OWL part of the study does not examine the date "
        "assertions or their order."
   
        in text
    )


def test_environment_captures_reasoner_provenance():
    payload = json.loads(
        (EXPERIMENTS / "environment.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["packages"]["rdflib"]
    assert payload["packages"]["pyshacl"]
    assert payload["packages"]["owlready2"]
    assert payload["java"]["version_output"]
    assert payload["hermit_jars"]

    for jar in payload["hermit_jars"]:
        assert len(jar["sha256"]) == 64
        assert jar["size_bytes"] > 0


def test_manifest_records_50_cases_and_outputs():
    payload = json.loads(
        (EXPERIMENTS / "validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["selection"]["selection_size"] == 50
    assert payload["controlled_manifest_rows"] == 50
    assert payload["counts"]["controlled_output_files"] > 250
    assert payload["counts"]["result_files"] >= 8

    assert (
        payload["convergence_rule"]
        == "Converged when both the violation identity set and the asserted "
        "triple set are unchanged across two consecutive repair rounds."
    )


def test_manifest_contains_core_results():
    payload = json.loads(
        (EXPERIMENTS / "validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    by_path = {row["path"]: row for row in payload["results"]}

    required = {
        "results/controlled_symbolic_validation.jsonl",
        "results/controlled_grounding_validation.jsonl",
        "results/controlled_grounding_target_analysis.json",
        "results/grounding_judge_v3_calibration.jsonl",
        "results/grounding_judge_v3_heldout.jsonl",
    }

    assert required <= set(by_path)

    for path in required:
        assert len(by_path[path]["sha256"]) == 64


def test_frozen_grounding_identity_is_unchanged():
    manifest = json.loads(
        (EXPERIMENTS / "validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    grounding = manifest["grounding_identity"]

    assert grounding["judge_version"] == "v3"
    assert grounding["model"] == "qwen2.5:7b-instruct-q4_K_M"
    assert grounding["model_digest"] == (
        "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
    )
    assert grounding["prompt_sha256"] == (
        "7fa341845f64e6c2b7079026bc4567e33f0ef3910c7e02f3efc654559223e73b"
    )
