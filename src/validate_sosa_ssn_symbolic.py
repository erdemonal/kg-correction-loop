from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from owlready2 import OwlReadyInconsistentOntologyError, World, sync_reasoner
from pyshacl import validate
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, SH, XSD

from src.build_sosa_ssn_confirmatory_cases import (
    case_data_graph,
    case_shapes_graph,
    merge_graphs,
    profile_graph,
    read_json,
    read_jsonl,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments" / "sosa_ssn_symbolic_validation_spec.json"
SOSA = "http://www.w3.org/ns/sosa/"


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
        raise RuntimeError(f"missing locked input: {path_value}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"input hash mismatch for {path_value}: expected {expected_sha256}, got {actual}"
        )
    return path


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def load_ontology(spec: dict) -> Graph:
    root = repository_path(spec["ontology"]["vendored_core_root"])
    expected = spec["ontology"]["module_sha256"]
    actual_names = {path.name for path in root.glob("*.ttl")}
    if actual_names != set(expected):
        raise RuntimeError(
            "vendored ontology module set changed: "
            f"expected {sorted(expected)}, got {sorted(actual_names)}"
        )
    graph = Graph()
    for name, digest in sorted(expected.items()):
        path = root / name
        if sha256_file(path) != digest:
            raise RuntimeError(f"ontology hash mismatch: {path.relative_to(ROOT)}")
        graph.parse(path, format="turtle")
    assert_required_axioms(graph)
    return graph


def assert_required_axioms(graph: Graph) -> None:
    for name in ("hasResult", "hasSimpleResult"):
        triple = (URIRef(SOSA + name), RDF.type, OWL.FunctionalProperty)
        if triple not in graph:
            raise RuntimeError(f"required functional-property axiom missing: sosa:{name}")

    classes = (
        "ActuationCollection",
        "ObservationCollection",
        "SampleCollection",
        "SamplingCollection",
    )
    for index, left in enumerate(classes):
        for right in classes[index + 1 :]:
            forward = (URIRef(SOSA + left), OWL.disjointWith, URIRef(SOSA + right))
            reverse = (forward[2], OWL.disjointWith, forward[0])
            if forward not in graph and reverse not in graph:
                raise RuntimeError(
                    f"required collection disjointness missing: {left}, {right}"
                )


def shacl_outcome(data_graph: Graph, shapes_graph: Graph) -> dict:
    conforms, report_graph, _ = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        advanced=True,
    )
    results = set(report_graph.subjects(RDF.type, SH.ValidationResult))
    components = sorted(
        {
            str(component)
            for result in results
            for component in report_graph.objects(result, SH.sourceConstraintComponent)
        }
    )
    return {
        "conforms": bool(conforms),
        "result_count": len(results),
        "constraint_components": components,
    }


def hermit_compatible_graph(graph: Graph) -> tuple[Graph, dict[str, int]]:
    output = Graph()
    removed = {"owl_imports": 0, "xsd_date": 0}
    for subject, predicate, obj in graph:
        if predicate == OWL.imports:
            removed["owl_imports"] += 1
            continue
        if obj == XSD.date or (
            isinstance(obj, Literal) and obj.datatype == XSD.date
        ):
            removed["xsd_date"] += 1
            continue
        output.add((subject, predicate, obj))
    return output, removed


def owl_consistent(graph: Graph) -> tuple[bool, dict[str, int]]:
    compatible, removed = hermit_compatible_graph(graph)
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "sosa-ssn-preflight.rdf"
        compatible.serialize(destination=str(path), format="xml")
        world = World()
        ontology = world.get_ontology(path.as_uri()).load()
        try:
            sync_reasoner([ontology], debug=0)
        except OwlReadyInconsistentOntologyError:
            return False, removed
    return True, removed


def union_graph(ontology: Graph, graphs: list[Graph]) -> Graph:
    return merge_graphs(ontology, *graphs)


def expected_for(case: dict, variant: str, spec: dict) -> dict:
    if variant == "clean":
        return spec["expected_symbolic"]["clean"]
    return spec["expected_symbolic"]["injected"][case["condition"]]


def validate_shacl(cases: list[dict], spec: dict) -> dict[str, dict]:
    profile = profile_graph(read_json(repository_path(spec["inputs"]["case_spec"])))
    output = {}
    for index, case in enumerate(cases, start=1):
        shapes = merge_graphs(profile, case_shapes_graph(case))
        clean = shacl_outcome(case_data_graph(case, "clean"), shapes)
        injected = shacl_outcome(case_data_graph(case, "injected"), shapes)
        allowed = set(
            spec["allowed_injected_shacl_components"][case["condition"]]
        )
        observed = set(injected["constraint_components"])
        if allowed and not allowed & observed:
            raise RuntimeError(
                f"{case['case_id']}: injected SHACL report lacks an allowed "
                f"component from {sorted(allowed)}; observed {sorted(observed)}"
            )
        output[case["case_id"]] = {"clean": clean, "injected": injected}
        if index % 30 == 0 or index == len(cases):
            print(f"SHACL: {index}/{len(cases)} cases")
    return output


def validate_owl(cases: list[dict], spec: dict, ontology: Graph) -> dict[str, dict]:
    output = {case["case_id"]: {} for case in cases}

    clean_union = union_graph(
        ontology, [case_data_graph(case, "clean") for case in cases]
    )
    clean_consistent, clean_removed = owl_consistent(clean_union)
    if not clean_consistent:
        raise RuntimeError("the union of all 180 clean graphs is OWL inconsistent")
    for case in cases:
        output[case["case_id"]]["clean"] = {
            "consistent": True,
            "strategy": "consistent_union_entails_consistent_subgraph",
        }
    print("OWL: clean union consistent (180/180 cases)")

    negative_conditions = set(
        spec["validators"]["owl_consistency"]["individual_negative_conditions"]
    )
    positive = [case for case in cases if case["condition"] not in negative_conditions]
    positive_union = union_graph(
        ontology, [case_data_graph(case, "injected") for case in positive]
    )
    positive_consistent, positive_removed = owl_consistent(positive_union)
    if not positive_consistent:
        raise RuntimeError("the union of injected OWL-positive graphs is inconsistent")
    for case in positive:
        output[case["case_id"]]["injected"] = {
            "consistent": True,
            "strategy": "consistent_union_entails_consistent_subgraph",
        }
    print(f"OWL: injected positive union consistent ({len(positive)}/{len(positive)} cases)")

    negative = [case for case in cases if case["condition"] in negative_conditions]
    negative.sort(key=lambda row: (row["condition"], row["case_id"]))
    for index, case in enumerate(negative, start=1):
        graph = merge_graphs(ontology, case_data_graph(case, "injected"))
        consistent, removed = owl_consistent(graph)
        output[case["case_id"]]["injected"] = {
            "consistent": consistent,
            "strategy": "individual_reasoner_run",
            "hermit_compatibility_removed": removed,
        }
        print(f"OWL negative: {index}/{len(negative)} {case['case_id']}")

    return {
        "by_case": output,
        "union_metadata": {
            "clean_case_count": len(cases),
            "clean_hermit_compatibility_removed": clean_removed,
            "injected_positive_case_count": len(positive),
            "injected_positive_hermit_compatibility_removed": positive_removed,
            "injected_negative_individual_case_count": len(negative),
        },
    }


def build_results(cases: list[dict], spec: dict, shacl: dict, owl: dict) -> list[dict]:
    results = []
    for case in cases:
        observed = {
            "clean": {
                "raw_shacl": shacl[case["case_id"]]["clean"]["conforms"],
                "owl_consistent": owl["by_case"][case["case_id"]]["clean"]["consistent"],
            },
            "injected": {
                "raw_shacl": shacl[case["case_id"]]["injected"]["conforms"],
                "owl_consistent": owl["by_case"][case["case_id"]]["injected"]["consistent"],
            },
        }
        mismatches = []
        for variant in ("clean", "injected"):
            expected = expected_for(case, variant, spec)
            for validator_name, expected_value in expected.items():
                if observed[variant][validator_name] != expected_value:
                    mismatches.append(
                        {
                            "variant": variant,
                            "validator": validator_name,
                            "expected": expected_value,
                            "observed": observed[variant][validator_name],
                        }
                    )
        results.append(
            {
                "case_id": case["case_id"],
                "condition": case["condition"],
                "source_family": case["source_family"],
                "observed": observed,
                "shacl_evidence": shacl[case["case_id"]],
                "owl_evidence": owl["by_case"][case["case_id"]],
                "mismatches": mismatches,
            }
        )
    return sorted(results, key=lambda row: (row["condition"], row["case_id"]))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def validate_inputs(spec: dict) -> list[dict]:
    for name in (
        "cases",
        "case_manifest",
        "case_spec",
        "application_profile",
        "axiom_inventory",
    ):
        verify_file(spec["inputs"][name], spec["inputs"][f"{name}_sha256"])
    cases = read_jsonl(repository_path(spec["inputs"]["cases"]))
    if len(cases) != 180 or len({row["case_id"] for row in cases}) != 180:
        raise RuntimeError("confirmatory denominator or case IDs changed")
    if Counter(row["condition"] for row in cases) != {
        condition: 30 for condition in spec["expected_symbolic"]["injected"]
    }:
        raise RuntimeError("condition denominators changed")
    return cases


def run(spec_path: Path = DEFAULT_SPEC) -> dict:
    spec = read_json(spec_path)
    scope = spec["execution_scope"]
    if not scope["runs_shacl"] or not scope["runs_reasoner"]:
        raise RuntimeError("symbolic validation execution scope changed")
    forbidden = (
        "runs_extractor",
        "runs_repair_model",
        "runs_grounding_assessor",
        "modifies_preliminary_results",
        "confirmatory_model_outcomes",
    )
    if any(scope[name] for name in forbidden):
        raise RuntimeError("forbidden experimental execution enabled")

    cases = validate_inputs(spec)
    ontology = load_ontology(spec)
    shacl = validate_shacl(cases, spec)
    owl = validate_owl(cases, spec, ontology)
    results = build_results(cases, spec, shacl, owl)
    mismatches = [row for row in results if row["mismatches"]]

    results_path = repository_path(spec["outputs"]["results"])
    manifest_path = repository_path(spec["outputs"]["manifest"])
    write_jsonl(results_path, results)
    manifest = {
        "version": 1,
        "spec_sha256": sha256_file(spec_path),
        "inputs": {
            name: {
                "path": spec["inputs"][name],
                "sha256": spec["inputs"][f"{name}_sha256"],
            }
            for name in (
                "cases",
                "case_manifest",
                "case_spec",
                "application_profile",
                "axiom_inventory",
            )
        },
        "ontology": {
            "pinned_commit": spec["ontology"]["pinned_commit"],
            "module_sha256": spec["ontology"]["module_sha256"],
            "parsed_triple_count": len(ontology),
        },
        "environment": {
            "rdflib": package_version("rdflib"),
            "pyshacl": package_version("pyshacl"),
            "owlready2": package_version("owlready2"),
            "reasoner": "HermiT through Owlready2",
        },
        "results": {
            "path": spec["outputs"]["results"],
            "sha256": sha256_file(results_path),
            "case_count": len(results),
            "by_condition": dict(sorted(Counter(row["condition"] for row in results).items())),
            "mismatch_case_count": len(mismatches),
        },
        "owl_strategy": owl["union_metadata"],
        "execution_scope": scope,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mismatches:
        raise RuntimeError(f"symbolic oracle mismatch in {len(mismatches)} cases")
    return manifest


def main() -> None:
    manifest = run()
    print(f"cases: {manifest['results']['case_count']}")
    print(f"mismatch cases: {manifest['results']['mismatch_case_count']}")
    print("SHACL and OWL oracle preflight passed.")
    print("No extractor, repair model, or grounding assessor was executed.")


if __name__ == "__main__":
    main()
