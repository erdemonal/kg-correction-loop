#!/usr/bin/env python3
"""Build clean and single fault SOSA and SSN confirmatory case records offline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments" / "sosa_ssn_case_spec.json"
SOSA = "http://www.w3.org/ns/sosa/"
KCL = "https://github.com/erdemonal/kg-correction-loop#"
OWL = "http://www.w3.org/2002/07/owl#"
CASE_BASE = "https://github.com/erdemonal/kg-correction-loop#sosa-case/"

CLASS_IRIS = {
    "Observation": f"{SOSA}Observation",
    "ObservationCollection": f"{SOSA}ObservationCollection",
    "SampleCollection": f"{SOSA}SampleCollection",
    "SamplingCollection": f"{SOSA}SamplingCollection",
    "ActuationCollection": f"{SOSA}ActuationCollection",
    "Actuation": f"{SOSA}Actuation",
    "Sample": f"{SOSA}Sample",
    "MaterialSample": f"{SOSA}MaterialSample",
    "FeatureOfInterest": f"{SOSA}FeatureOfInterest",
    "Property": f"{SOSA}Property",
    "Sensor": f"{SOSA}Sensor",
    "Actuator": f"{SOSA}Actuator",
    "USGSDailyObservation": f"{KCL}USGSDailyObservation",
}

PREDICATE_IRIS = {
    "hasMember": f"{SOSA}hasMember",
    "hasFeatureOfInterest": f"{SOSA}hasFeatureOfInterest",
    "hasUltimateFeatureOfInterest": f"{SOSA}hasUltimateFeatureOfInterest",
    "observedProperty": f"{SOSA}observedProperty",
    "hasSimpleResult": f"{SOSA}hasSimpleResult",
    "hasResult": f"{SOSA}hasResult",
    "resultTime": f"{SOSA}resultTime",
    "phenomenonTime": f"{SOSA}phenomenonTime",
    "startTime": f"{SOSA}startTime",
    "endTime": f"{SOSA}endTime",
    "madeBySensor": f"{SOSA}madeBySensor",
    "madeByActuator": f"{SOSA}madeByActuator",
    "actsOnProperty": f"{SOSA}actsOnProperty",
    "isSampleOf": f"{SOSA}isSampleOf",
    "collectionStart": f"{KCL}collectionStart",
    "collectionEnd": f"{KCL}collectionEnd",
    "resultValue": f"{KCL}resultValue",
    "resultUnit": f"{KCL}resultUnit",
    "differentFrom": f"{OWL}differentFrom",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        raise RuntimeError(f"missing frozen input: {path_value}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"input hash mismatch for {path_value}: expected {expected_sha256}, got {actual}"
        )
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


def triple(subject: str, predicate: str, obj: str, object_kind: str = "entity") -> dict:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": str(obj),
        "object_kind": object_kind,
    }


def triple_from_array(row: list) -> dict:
    if not isinstance(row, list) or len(row) != 4:
        raise RuntimeError(f"invalid registry triple: {row!r}")
    return triple(*row)


def triple_key(row: dict) -> tuple:
    return (
        row["subject"],
        row["predicate"],
        row["object"],
        row["object_kind"],
    )


def validate_triple(row: dict, spec: dict) -> None:
    if set(row) != {"subject", "predicate", "object", "object_kind"}:
        raise RuntimeError(f"invalid triple fields: {row}")
    if not all(isinstance(row[key], str) and row[key] for key in row):
        raise RuntimeError(f"empty or non-string triple value: {row}")
    if row["predicate"] not in spec["relations"]:
        raise RuntimeError(f"unknown relation: {row['predicate']}")
    allowed_kinds = {"entity", *spec["literal_kinds"]}
    if row["object_kind"] not in allowed_kinds:
        raise RuntimeError(f"unknown object kind: {row['object_kind']}")
    if row["predicate"] == "type" and row["object"] not in CLASS_IRIS:
        raise RuntimeError(f"unknown class: {row['object']}")


def entity_uri(case_id: str, label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
    return f"{CASE_BASE}{quote(case_id, safe='')}/{quote(label, safe='')[:96]}-{digest}"


def object_term(case_id: str, row: dict):
    from rdflib import Literal, URIRef
    from rdflib.namespace import XSD

    value = row["object"]
    kind = row["object_kind"]
    if kind == "entity":
        return URIRef(entity_uri(case_id, value))
    if kind == "string":
        return Literal(value)
    if kind == "decimal":
        try:
            Decimal(value)
        except InvalidOperation as exc:
            raise RuntimeError(f"invalid decimal literal: {value}") from exc
        return Literal(value, datatype=XSD.decimal)
    if kind == "boolean":
        if value not in {"true", "false"}:
            raise RuntimeError(f"invalid boolean literal: {value}")
        return Literal(value == "true", datatype=XSD.boolean)
    if kind == "date":
        return Literal(value, datatype=XSD.date)
    if kind == "datetime":
        return Literal(value, datatype=XSD.dateTime)
    raise RuntimeError(f"unsupported object kind: {kind}")


def triples_to_graph(case_id: str, rows: list[dict]):
    from rdflib import Graph, URIRef
    from rdflib.namespace import RDF

    graph = Graph()
    for row in rows:
        subject = URIRef(entity_uri(case_id, row["subject"]))
        if row["predicate"] == "type":
            graph.add((subject, RDF.type, URIRef(CLASS_IRIS[row["object"]])))
            continue
        predicate = URIRef(PREDICATE_IRIS[row["predicate"]])
        graph.add((subject, predicate, object_term(case_id, row)))
    return graph


def profile_graph(spec: dict):
    from rdflib import Graph

    profile = spec["profile"]
    path = verify_file(profile["path"], profile["sha256"])
    graph = Graph()
    graph.parse(path, format="turtle")
    return graph


def case_shapes_graph(case: dict):
    from rdflib import Graph, Literal, URIRef
    from rdflib.namespace import RDF, SH

    graph = Graph()
    constraint = case.get("case_constraint")
    if constraint is None:
        return graph
    if constraint != {
        "kind": "min_count",
        "focus": constraint.get("focus"),
        "predicate": constraint.get("predicate"),
        "value": 1,
    }:
        raise RuntimeError(f"unsupported case constraint: {constraint}")
    focus = URIRef(entity_uri(case["case_id"], constraint["focus"]))
    shape = URIRef(f"{focus}/minimum-count-shape")
    property_shape = URIRef(f"{focus}/minimum-count-property")
    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, SH.targetNode, focus))
    graph.add((shape, SH.property, property_shape))
    graph.add((property_shape, SH.path, URIRef(PREDICATE_IRIS[constraint["predicate"]])))
    graph.add((property_shape, SH.minCount, Literal(1)))
    return graph


def merge_graphs(*graphs):
    from rdflib import Graph

    output = Graph()
    for graph in graphs:
        for item in graph:
            output.add(item)
    return output


def case_data_graph(case: dict, variant: str):
    if variant not in {"clean", "injected"}:
        raise ValueError("variant must be clean or injected")
    content = case[f"{variant}_content_triples"]
    return triples_to_graph(case["case_id"], content + case["scaffold_triples"])


def alter_decimal(value: str) -> str:
    number = Decimal(value)
    altered = number + Decimal("1")
    return format(altered, "f")


def apply_modification(clean: list[dict], modification: dict) -> list[dict]:
    current = {triple_key(row): row for row in clean}
    if len(current) != len(clean):
        raise RuntimeError("clean graph contains duplicate triples")
    for row in modification["removed"]:
        key = triple_key(row)
        if key not in current:
            raise RuntimeError(f"cannot remove absent triple: {row}")
        del current[key]
    for row in modification["added"]:
        key = triple_key(row)
        if key in current:
            raise RuntimeError(f"cannot add existing triple: {row}")
        current[key] = row
    return [current[key] for key in sorted(current)]


def usgs_clean(unit: dict) -> tuple[list[dict], list[dict], dict]:
    record = unit["normalized_record"]
    token = record["time_series_id"][:12]
    observation = f"observation_{token}"
    collection = f"daily_collection_{token}"
    site = record["monitoring_location_id"]
    prop = f"parameter_{record['parameter_code']}"
    day = record["time"]
    clean = [
        triple(collection, "type", "ObservationCollection"),
        triple(observation, "type", "Observation"),
        triple(site, "type", "FeatureOfInterest"),
        triple(prop, "type", "Property"),
        triple(collection, "hasMember", observation),
        triple(collection, "hasFeatureOfInterest", site),
        triple(observation, "hasFeatureOfInterest", site),
        triple(collection, "observedProperty", prop),
        triple(observation, "observedProperty", prop),
        triple(observation, "hasSimpleResult", record["value"], "decimal"),
        triple(observation, "resultUnit", record["unit_of_measure"], "string"),
        triple(observation, "phenomenonTime", day, "date"),
        triple(observation, "resultTime", day, "date"),
        triple(collection, "collectionStart", day, "date"),
        triple(collection, "collectionEnd", day, "date"),
    ]
    scaffold = [triple(observation, "type", "USGSDailyObservation")]
    entities = {
        "observation": observation,
        "collection": collection,
        "site": site,
        "property": prop,
    }
    return clean, scaffold, entities


def usgs_modification(condition: str, clean: list[dict], entities: dict) -> tuple[dict, dict | None]:
    observation = entities["observation"]
    collection = entities["collection"]
    site = entities["site"]
    prop = entities["property"]
    by_key = {triple_key(row): row for row in clean}
    case_constraint = None

    if condition == "disjointness":
        added = [triple(collection, "type", "SampleCollection")]
        removed = []
        operation = "add"
    elif condition == "functional_property_conflict":
        old = next(
            row
            for row in clean
            if row["subject"] == observation and row["predicate"] == "hasSimpleResult"
        )
        added = [triple(observation, "hasSimpleResult", alter_decimal(old["object"]), "decimal")]
        removed = []
        operation = "add"
    elif condition == "domain_range":
        old = triple(observation, "hasFeatureOfInterest", site)
        added = [triple(observation, "hasFeatureOfInterest", prop)]
        removed = [by_key[triple_key(old)]]
        operation = "replace"
    elif condition == "cardinality":
        old = triple(observation, "observedProperty", prop)
        added = []
        removed = [by_key[triple_key(old)]]
        operation = "remove"
    elif condition == "temporal":
        old = next(
            row
            for row in clean
            if row["subject"] == observation and row["predicate"] == "phenomenonTime"
        )
        outside = (date.fromisoformat(old["object"]) + timedelta(days=1)).isoformat()
        added = [triple(observation, "phenomenonTime", outside, "date")]
        removed = [old]
        operation = "replace"
    elif condition == "grounding":
        old = next(
            row
            for row in clean
            if row["subject"] == observation and row["predicate"] == "hasSimpleResult"
        )
        added = [triple(observation, "hasSimpleResult", alter_decimal(old["object"]), "decimal")]
        removed = [old]
        operation = "replace"
    else:
        raise RuntimeError(f"unsupported condition: {condition}")

    return {"operation": operation, "added": added, "removed": removed}, case_constraint


def usgs_case_text(unit: dict, entities: dict) -> str:
    return (
        f"{unit['source_text']} In the record derived SOSA representation, "
        f"{entities['observation']} is an Observation and a member of "
        f"ObservationCollection {entities['collection']} for that day. "
        f"{entities['site']} is the FeatureOfInterest and "
        f"{entities['property']} is the observed Property."
    )


def build_usgs_case(selection: dict, unit: dict, spec: dict) -> dict:
    clean, scaffold, entities = usgs_clean(unit)
    modification, constraint = usgs_modification(
        selection["condition"], clean, entities
    )
    injected = apply_modification(clean, modification)
    return base_case(
        selection,
        unit,
        spec,
        source_text=usgs_case_text(unit, entities),
        clean=clean,
        scaffold=scaffold,
        modification=modification,
        injected=injected,
        case_constraint=constraint,
    )


def build_w3c_case(selection: dict, unit: dict, entry: dict, spec: dict) -> dict:
    if entry["condition"] != selection["condition"]:
        raise RuntimeError(
            f"W3C registry condition mismatch for {selection['source_unit_id']}"
        )
    clean = [triple_from_array(row) for row in entry["clean_triples"]]
    scaffold = [triple_from_array(row) for row in entry.get("scaffold_triples", [])]
    modification = {
        "operation": entry["injection"]["operation"],
        "added": [triple_from_array(row) for row in entry["injection"]["added"]],
        "removed": [triple_from_array(row) for row in entry["injection"]["removed"]],
    }
    injected = apply_modification(clean, modification)
    return base_case(
        selection,
        unit,
        spec,
        source_text=unit["source_text"],
        clean=clean,
        scaffold=scaffold,
        modification=modification,
        injected=injected,
        case_constraint=entry.get("case_constraint"),
    )


def base_case(
    selection: dict,
    unit: dict,
    spec: dict,
    *,
    source_text: str,
    clean: list[dict],
    scaffold: list[dict],
    modification: dict,
    injected: list[dict],
    case_constraint: dict | None,
) -> dict:
    for row in clean + scaffold + modification["added"] + modification["removed"]:
        validate_triple(row, spec)
    if not modification["added"] and not modification["removed"]:
        raise RuntimeError("empty controlled modification")
    condition = selection["condition"]
    clean_expected = spec["expected_symbolic"]["clean"]
    injected_expected = spec["expected_symbolic"]["injected"][condition]
    return {
        "case_id": selection["case_id"],
        "condition": condition,
        "source_unit_id": selection["source_unit_id"],
        "source_family": selection["source_family"],
        "scenario_family": selection["scenario_family"],
        "source_text": source_text,
        "source_text_sha256": sha256_text(source_text),
        "source_unit_text_sha256": unit["source_text_sha256"],
        "allowed_relations": list(spec["relations"]),
        "clean_content_triples": sorted(clean, key=triple_key),
        "scaffold_triples": sorted(scaffold, key=triple_key),
        "injected_content_triples": injected,
        "primary_modification": modification,
        "case_constraint": case_constraint,
        "expected_symbolic": {
            "clean": dict(clean_expected),
            "injected": dict(injected_expected),
        },
        "grounding_scope": "content_triples_only",
        "clean_reference_scope": "clean_content_triples_only",
    }


def unique_index(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    output = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{label}: missing {key}")
        if value in output:
            raise RuntimeError(f"{label}: duplicate {key}: {value}")
        output[value] = row
    return output


def validate_cases(cases: list[dict], spec: dict) -> None:
    if len(cases) != 180:
        raise RuntimeError(f"expected 180 cases, found {len(cases)}")
    if len({case["case_id"] for case in cases}) != 180:
        raise RuntimeError("duplicate case ID")
    if len({case["source_unit_id"] for case in cases}) != 180:
        raise RuntimeError("source unit reused")
    if Counter(case["condition"] for case in cases) != {
        condition: 30 for condition in spec["expected_symbolic"]["injected"]
    }:
        raise RuntimeError("condition denominators changed")
    for case in cases:
        clean = {triple_key(row) for row in case["clean_content_triples"]}
        injected = {triple_key(row) for row in case["injected_content_triples"]}
        added = {triple_key(row) for row in case["primary_modification"]["added"]}
        removed = {triple_key(row) for row in case["primary_modification"]["removed"]}
        if injected != (clean - removed) | added:
            raise RuntimeError(f"modification does not reproduce {case['case_id']}")
        if added & clean or not removed <= clean:
            raise RuntimeError(f"invalid added/removed sets in {case['case_id']}")


def build(spec_path: Path = DEFAULT_SPEC) -> dict:
    spec = read_json(spec_path)
    if any(spec["execution"].values()):
        raise RuntimeError("case construction execution guard changed")
    selection_path = verify_file(
        spec["inputs"]["selection"], spec["inputs"]["selection_sha256"]
    )
    units_path = verify_file(
        spec["inputs"]["source_units"], spec["inputs"]["source_units_sha256"]
    )
    verify_file(
        spec["inputs"]["sampling_manifest"],
        spec["inputs"]["sampling_manifest_sha256"],
    )
    verify_file(spec["profile"]["path"], spec["profile"]["sha256"])
    registry_path = verify_file(
        spec["w3c_registry"]["path"], spec["w3c_registry"]["sha256"]
    )
    registry = read_json(registry_path)["entries"]
    selection = read_jsonl(selection_path)
    units = unique_index(read_jsonl(units_path), "source_unit_id", "source units")

    cases = []
    used_registry = set()
    for selected in selection:
        unit = units.get(selected["source_unit_id"])
        if unit is None:
            raise RuntimeError(f"selected source unit missing: {selected['source_unit_id']}")
        if unit["source_family"] != selected["source_family"]:
            raise RuntimeError(f"source family mismatch: {selected['source_unit_id']}")
        if unit["source_family"] == "usgs_daily":
            case = build_usgs_case(selected, unit, spec)
        elif unit["source_family"] == "w3c_examples":
            entry = registry.get(unit["source_unit_id"])
            if entry is None:
                raise RuntimeError(f"W3C registry entry missing: {unit['source_unit_id']}")
            used_registry.add(unit["source_unit_id"])
            case = build_w3c_case(selected, unit, entry, spec)
        else:
            raise RuntimeError(f"unsupported source family: {unit['source_family']}")
        cases.append(case)

    if used_registry != set(registry):
        raise RuntimeError("W3C registry and selected W3C units differ")
    cases.sort(key=lambda row: (row["condition"], row["case_id"]))
    validate_cases(cases, spec)

    cases_path = repository_path(spec["outputs"]["cases"])
    manifest_path = repository_path(spec["outputs"]["manifest"])
    write_jsonl(cases_path, cases)
    manifest = {
        "version": 1,
        "case_spec_sha256": sha256_file(spec_path),
        "sampling_commit": spec["sampling_commit"],
        "cases": {
            "path": spec["outputs"]["cases"],
            "sha256": sha256_file(cases_path),
            "count": len(cases),
            "by_condition": dict(sorted(Counter(row["condition"] for row in cases).items())),
            "by_source_family": dict(sorted(Counter(row["source_family"] for row in cases).items())),
        },
        "profile": spec["profile"],
        "w3c_registry": spec["w3c_registry"],
        "semantic_separation": spec["semantic_layers"],
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
    print(f"cases: {manifest['cases']['count']}")
    for condition, count in manifest["cases"]["by_condition"].items():
        print(f"  {condition}: {count}")
    print("Clean and injected case records were constructed offline.")
    print("No model, validator, reasoner, or grounding assessor was executed.")


if __name__ == "__main__":
    main()
