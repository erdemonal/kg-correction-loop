from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pyshacl import validate
from rdflib import BNode, Graph, URIRef
from rdflib.compare import to_canonical_graph
from rdflib.namespace import RDF, SH

from src import extract_text2kg, grounding_judge
from src.build_sosa_ssn_confirmatory_cases import (
    CLASS_IRIS,
    PREDICATE_IRIS,
    case_shapes_graph,
    entity_uri,
    merge_graphs,
    profile_graph,
    read_json,
    read_jsonl,
    sha256_file,
    triple,
    triples_to_graph,
)
from src.validate_sosa_ssn_symbolic import load_ontology, owl_consistent


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SPEC_PATH = ROOT / "experiments" / "sosa_ssn_confirmatory_runner_spec.json"
AUDIT_GATE_PATH = ROOT / "experiments" / "sosa_ssn_confirmatory_audit_gate.json"

ENTITY_PREDICATES = {
    "hasMember",
    "hasFeatureOfInterest",
    "hasUltimateFeatureOfInterest",
    "observedProperty",
    "hasResult",
    "madeBySensor",
    "madeByActuator",
    "actsOnProperty",
    "isSampleOf",
    "differentFrom",
}
TIME_PREDICATES = {
    "resultTime",
    "phenomenonTime",
    "startTime",
    "endTime",
    "collectionStart",
    "collectionEnd",
}


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repository_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {value}") from exc
    return path


def git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_changed_since(commit: str) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", f"{commit}..HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def git_tracked_dirty_paths() -> set[str]:
    paths = set()
    for args in (
        ["git", "-C", str(ROOT), "diff", "--name-only"],
        ["git", "-C", str(ROOT), "diff", "--cached", "--name-only"],
    ):
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        paths.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return paths


def load_runner_spec() -> dict:
    spec = read_json(RUNNER_SPEC_PATH)
    if spec.get("status") != "prepared_for_pre_run_audit":
        raise RuntimeError("confirmatory runner spec is not frozen for audit")
    for item in spec["inputs"].values():
        path = repository_path(item["path"])
        if not path.is_file():
            raise RuntimeError(f"missing frozen runner input: {item['path']}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(
                f"runner input hash mismatch for {item['path']}: "
                f"expected {item['sha256']}, got {actual}"
            )
    return spec


def load_cases(spec: dict | None = None) -> list[dict]:
    if spec is None:
        spec = load_runner_spec()
    rows = read_jsonl(repository_path(spec["inputs"]["cases"]["path"]))
    rows.sort(key=lambda row: (row["condition"], row["case_id"]))
    if len(rows) != spec["sample"]["cases"]:
        raise RuntimeError(f"expected 180 cases, found {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate confirmatory case id")
    if len({row["source_unit_id"] for row in rows}) != len(rows):
        raise RuntimeError("confirmatory source unit reused")
    expected = {
        condition: spec["sample"]["cases_per_condition"]
        for condition in spec["sample"]["conditions"]
    }
    if Counter(row["condition"] for row in rows) != expected:
        raise RuntimeError("confirmatory condition denominators changed")
    return rows


def case_index(cases: list[dict]) -> dict[str, dict]:
    output = {case["case_id"]: case for case in cases}
    if len(output) != len(cases):
        raise RuntimeError("duplicate case id")
    return output


def triple_tuple(row: dict) -> tuple[str, str, str]:
    return row["subject"], row["predicate"], row["object"]


def normalized_triples(rows) -> tuple[tuple[str, str, str], ...]:
    seen = set()
    output = []
    for row in rows:
        if isinstance(row, dict):
            value = triple_tuple(row)
        else:
            if (
                not isinstance(row, (list, tuple))
                or len(row) != 3
                or not all(isinstance(item, str) and item for item in row)
            ):
                raise ValueError(f"invalid content triple: {row!r}")
            value = tuple(row)
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def case_content(case: dict, state: str) -> tuple[tuple[str, str, str], ...]:
    if state == "clean":
        return normalized_triples(case["clean_content_triples"])
    if state == "injected":
        return normalized_triples(case["injected_content_triples"])
    raise ValueError("state must be clean or injected")


def primary_added(case: dict) -> tuple[tuple[str, str, str], ...]:
    return normalized_triples(case["primary_modification"]["added"])


def expected_target_grounding(case: dict, spec: dict) -> bool:
    return bool(spec["grounding"]["expected_target_error"][case["condition"]])


def reference_kind_map(case: dict) -> dict[tuple[str, str, str], str]:
    output = {}
    rows = case["clean_content_triples"] + case["injected_content_triples"]
    for row in rows:
        key = triple_tuple(row)
        previous = output.get(key)
        if previous is not None and previous != row["object_kind"]:
            raise RuntimeError(f"conflicting object kinds in {case['case_id']}: {key}")
        output[key] = row["object_kind"]
    return output


def infer_object_kind(case: dict, value: tuple[str, str, str]) -> str:
    known = reference_kind_map(case).get(value)
    if known is not None:
        return known
    _subject, predicate, obj = value
    if predicate == "type":
        if obj not in CLASS_IRIS:
            raise ValueError(f"unknown SOSA/SSN profile class: {obj}")
        return "entity"
    if predicate in ENTITY_PREDICATES:
        return "entity"
    if predicate in TIME_PREDICATES:
        try:
            if len(obj) == 10:
                date.fromisoformat(obj)
                return "date"
            datetime.fromisoformat(obj.replace("Z", "+00:00"))
            return "datetime"
        except ValueError as exc:
            raise ValueError(f"invalid ISO time value for {predicate}: {obj}") from exc
    if predicate == "resultUnit":
        return "string"
    if predicate in {"hasSimpleResult", "resultValue"}:
        if obj in {"true", "false"}:
            return "boolean"
        try:
            Decimal(obj)
            return "decimal"
        except InvalidOperation:
            return "string"
    raise ValueError(f"no deterministic object-kind rule for relation: {predicate}")


def rows_for_repair(case: dict, triples) -> list[dict]:
    output = []
    for value in normalized_triples(triples):
        output.append(triple(*value, object_kind=infer_object_kind(case, value)))
    return output


def repaired_symbolic_graph(case: dict, triples) -> Graph:
    rows = rows_for_repair(case, triples)
    return triples_to_graph(
        case["case_id"], rows + case["scaffold_triples"]
    )


def load_symbolic_context(spec: dict) -> tuple[dict, Graph, Graph]:
    case_spec = read_json(repository_path(spec["inputs"]["case_spec"]["path"]))
    symbolic_spec = read_json(
        repository_path(spec["inputs"]["symbolic_spec"]["path"])
    )
    profile = profile_graph(case_spec)
    ontology = load_ontology(symbolic_spec)
    return symbolic_spec, profile, ontology


def canonical_term(term) -> str | None:
    return None if term is None else term.n3()


def shacl_violation_identity(report: Graph, node) -> dict:
    fields = {
        "sourceConstraintComponent": report.value(node, SH.sourceConstraintComponent),
        "focusNode": report.value(node, SH.focusNode),
        "resultPath": report.value(node, SH.resultPath),
        "value": report.value(node, SH.value),
        "sourceShape": report.value(node, SH.sourceShape),
    }
    canonical = {key: canonical_term(value) for key, value in fields.items()}
    digest = sha256_bytes(canonical_json(canonical).encode("utf-8"))[:20]
    return {"violation_id": f"shacl:{digest}", "identity": canonical}


def shacl_results(data: Graph, shapes: Graph) -> dict:
    canonical_shapes = Graph()
    for item in to_canonical_graph(shapes):
        canonical_shapes.add(item)
    conforms, report, _ = validate(
        data_graph=data,
        shacl_graph=canonical_shapes,
        inference="none",
        advanced=True,
    )
    rows = []
    for node in report.subjects(RDF.type, SH.ValidationResult):
        identity = shacl_violation_identity(report, node)
        message = report.value(node, SH.resultMessage)
        component = report.value(node, SH.sourceConstraintComponent)
        rows.append(
            {
                **identity,
                "message": None if message is None else str(message),
                "focus_node": canonical_term(report.value(node, SH.focusNode)),
                "result_path": canonical_term(report.value(node, SH.resultPath)),
                "value": canonical_term(report.value(node, SH.value)),
                "constraint_component": None if component is None else str(component),
            }
        )
    rows.sort(key=lambda row: row["violation_id"])
    return {"conforms": bool(conforms), "violations": rows}


def display_maps(case: dict, triples) -> tuple[dict[str, str], dict[str, str]]:
    entity = {}
    labels = set()
    for row in (
        case["clean_content_triples"]
        + case["injected_content_triples"]
        + case["scaffold_triples"]
    ):
        labels.add(row["subject"])
        if row["object_kind"] == "entity" and row["predicate"] != "type":
            labels.add(row["object"])
    for subject, _predicate, obj in normalized_triples(triples):
        labels.add(subject)
        labels.add(obj)
    for label in labels:
        iri = entity_uri(case["case_id"], label)
        entity[str(iri)] = label
        entity[URIRef(iri).n3()] = label
    relation = {}
    for name, iri in PREDICATE_IRIS.items():
        relation[str(iri)] = name
        relation[URIRef(iri).n3()] = name
    return entity, relation


def readable(value: str | None, mapping: dict[str, str]) -> str | None:
    if value is None:
        return None
    if value in mapping:
        return mapping[value]
    if value.startswith("<") and value.endswith(">"):
        return mapping.get(value[1:-1], value)
    return value


def shacl_feedback(case: dict, triples, result: dict, *, initial: bool) -> list[dict]:
    entities, relations = display_maps(case, triples)
    output = []
    for violation in result["violations"]:
        output.append(
            {
                "validator": "raw_shacl",
                "violation_id": violation["violation_id"],
                "error_type": case["condition"] if initial else None,
                "focus": readable(violation["focus_node"], entities),
                "path": readable(violation["result_path"], relations),
                "message": violation["message"] or "A SHACL constraint is violated.",
            }
        )
    return output


def validate_symbolic_state(
    case: dict,
    triples,
    profile: Graph,
    ontology: Graph,
) -> dict:
    data = repaired_symbolic_graph(case, triples)
    shapes = merge_graphs(profile, case_shapes_graph(case))
    shacl = shacl_results(data, shapes)
    consistent, compatibility_removed = owl_consistent(merge_graphs(ontology, data))
    return {
        "shacl": shacl,
        "owl_consistent": consistent,
        "hermit_compatibility_removed": compatibility_removed,
    }


def target_focus(case: dict) -> str:
    modification = case["primary_modification"]
    source = modification["added"] or modification["removed"]
    if not source:
        raise RuntimeError(f"case has no controlled target: {case['case_id']}")
    return source[0]["subject"]


def target_resolved(
    case: dict,
    triples,
    symbolic: dict,
    symbolic_spec: dict,
    target_violation_ids: set[str] | None = None,
) -> bool:
    if case["condition"] == "grounding":
        current = set(normalized_triples(triples))
        return all(item not in current for item in primary_added(case))
    allowed = set(
        symbolic_spec["allowed_injected_shacl_components"][case["condition"]]
    )
    current_ids = {
        row["violation_id"] for row in symbolic["shacl"]["violations"]
    }
    if target_violation_ids is not None:
        return not bool(current_ids & target_violation_ids)
    for violation in symbolic["shacl"]["violations"]:
        if violation["constraint_component"] in allowed:
            return False
    return True


def owl_feedback(case: dict, consistent: bool, *, initial: bool) -> dict | None:
    if consistent:
        return None
    return {
        "validator": "owl_consistency",
        "violation_id": f"owl:inconsistent:{case['case_id']}",
        "error_type": case["condition"] if initial else None,
        "focus": target_focus(case) if initial else None,
        "path": None,
        "message": "The graph is logically inconsistent.",
    }


def grounding_identity(case_id: str, value: tuple[str, str, str]) -> str:
    encoded = canonical_json([case_id, *value]).encode("utf-8")
    return f"grounding:{sha256_bytes(encoded)[:20]}"


def grounding_feedback(case_id: str, value: tuple[str, str, str]) -> dict:
    subject, predicate, obj = value
    return {
        "validator": "grounding_v3",
        "violation_id": grounding_identity(case_id, value),
        "error_type": "grounding_error",
        "focus": f"{predicate}({subject}, {obj})",
        "path": predicate,
        "message": "The source text does not support this assertion.",
    }


def judgment_cache(grounding_row: dict) -> dict[tuple[str, str, str], dict]:
    output = {}
    for row in grounding_row.get("judgments", []):
        value = row.get("triple")
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(item, str) for item in value)
            or row.get("verdict") not in {"SUPPORTED", "UNSUPPORTED"}
        ):
            raise RuntimeError("invalid frozen confirmatory grounding judgment")
        key = tuple(value)
        if key in output:
            raise RuntimeError("duplicate frozen judgment within case")
        output[key] = row
    return output


def initial_grounding_feedback(case: dict, cache: dict) -> list[dict]:
    output = []
    for value in primary_added(case):
        judgment = cache.get(value)
        if judgment is None:
            raise RuntimeError(f"missing target grounding judgment: {case['case_id']} {value}")
        if judgment["verdict"] == "UNSUPPORTED":
            output.append(grounding_feedback(case["case_id"], value))
    return output


def clean_unsupported(case: dict, cache: dict) -> set[tuple[str, str, str]]:
    return {
        value
        for value in case_content(case, "clean")
        if cache[value]["verdict"] == "UNSUPPORTED"
    }


def later_grounding_feedback(
    case: dict,
    triples,
    cache: dict,
    baseline_unsupported: set[tuple[str, str, str]],
    template: str,
) -> tuple[list[dict], list[dict]]:
    judgments = []
    for value in normalized_triples(triples):
        row = cache.get(value)
        if row is None:
            judged = grounding_judge.judge_triple(
                case["source_text"], list(value), template=template
            )
            row = {**judged, "source": "repair_round"}
            cache[value] = row
        judgments.append(row)
    feedback = [
        grounding_feedback(case["case_id"], tuple(row["triple"]))
        for row in judgments
        if row["verdict"] == "UNSUPPORTED"
        and tuple(row["triple"]) not in baseline_unsupported
    ]
    return feedback, judgments


def format_triple(value: tuple[str, str, str]) -> str:
    subject, predicate, obj = value
    return f"{predicate}({subject}, {obj})"


def render_repair_prompt(
    template: str,
    case: dict,
    triples,
    feedback: list[dict],
) -> str:
    graph = "\n".join(format_triple(row) for row in normalized_triples(triples))
    if not graph:
        graph = "(empty graph)"
    relations = "\n".join(f"- {name}" for name in sorted(case["allowed_relations"]))
    ordered_feedback = sorted(
        feedback, key=lambda row: (row["validator"], row["violation_id"])
    )
    return template.format(
        source_text=case["source_text"],
        allowed_relations=relations,
        current_graph=graph,
        feedback=json.dumps(ordered_feedback, ensure_ascii=False, indent=2, sort_keys=True),
    )


def parse_repair_response(text: str, allowed_relations) -> dict:
    allowed = set(allowed_relations)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "failure": "empty_output", "triples": [], "details": None}
    parsed = []
    for line_number, line in enumerate(lines, start=1):
        stripped = extract_text2kg.strip_prefix(line)
        call = extract_text2kg.extract_call(stripped)
        if call is None:
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {"line_number": line_number, "line": line},
            }
        open_index = stripped.find("(")
        depth = 0
        close_index = None
        for index in range(open_index, len(stripped)):
            if stripped[index] == "(":
                depth += 1
            elif stripped[index] == ")":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        if close_index is None or stripped[close_index + 1 :].strip():
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {"line_number": line_number, "line": line},
            }
        relation, arguments = call
        pair = extract_text2kg.split_arguments(arguments)
        if pair is None:
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {"line_number": line_number, "line": line},
            }
        subject, obj = pair
        relation = "_".join(relation.replace("\\_", "_").strip().split())
        subject, obj = subject.strip(), obj.strip()
        if not relation or not subject or not obj:
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {"line_number": line_number, "line": line},
            }
        if relation not in allowed:
            return {
                "ok": False,
                "failure": "relation_outside_allowed_set",
                "triples": [],
                "details": {"line_number": line_number, "relation": relation, "line": line},
            }
        parsed.append((subject, relation, obj))
    return {
        "ok": True,
        "failure": None,
        "triples": [list(row) for row in normalized_triples(parsed)],
        "details": None,
    }


def validate_parsed_kinds(case: dict, parsed: dict) -> dict:
    if not parsed["ok"]:
        return parsed
    try:
        rows_for_repair(case, parsed["triples"])
    except (ValueError, RuntimeError) as exc:
        return {
            "ok": False,
            "failure": "invalid_object_kind",
            "triples": [],
            "details": {"message": str(exc)},
        }
    return parsed


def verify_model_metadata(spec: dict) -> tuple[dict, dict]:
    version = extract_text2kg.api("/api/version")["version"]
    models = extract_text2kg.api("/api/tags")["models"]
    output = []
    for role in ("repair", "grounding"):
        expected = spec["models"][role]
        found = next(
            (
                row
                for row in models
                if row.get("name") == expected["name"]
                or row.get("model") == expected["name"]
            ),
            None,
        )
        if found is None:
            raise RuntimeError(f"missing Ollama model: {expected['name']}")
        if found.get("digest") != expected["digest"]:
            raise RuntimeError(
                f"{role} model digest mismatch: expected {expected['digest']}, "
                f"got {found.get('digest')}"
            )
        output.append(
            {
                "role": role,
                "name": expected["name"],
                "digest": found.get("digest"),
                "options": expected["options"],
                "ollama_version": version,
            }
        )
    return output[0], output[1]


def require_accepted_audit_gate() -> dict:
    gate = read_json(AUDIT_GATE_PATH)
    if gate.get("execution_allowed") is not True or gate.get("status") != "accepted":
        raise RuntimeError("confirmatory execution is blocked pending accepted pre-run audit")
    if gate.get("verdict") != "A":
        raise RuntimeError("audit gate does not record verdict A")
    commit = gate.get("audited_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("audit gate has no full audited commit SHA")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("audited commit is not an ancestor of the execution commit")
    allowed = {
        "experiments/sosa_ssn_confirmatory_audit_gate.json",
        "docs/audit_register.md",
    }
    if set(gate.get("allowed_post_audit_changes", [])) != allowed:
        raise RuntimeError("audit gate post-audit allowlist changed")
    changed = git_changed_since(commit)
    if not changed <= allowed:
        raise RuntimeError(
            "unaudited files changed after the audited commit: "
            + ", ".join(sorted(changed - allowed))
        )
    dirty = git_tracked_dirty_paths()
    if dirty:
        raise RuntimeError(
            "tracked working-tree changes are present at execution: "
            + ", ".join(sorted(dirty))
        )
    return gate


def load_complete_jsonl(path: Path, valid_ids: set[str]) -> list[dict]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    ids = []
    for row in rows:
        case_id = row.get("case_id")
        if case_id not in valid_ids:
            raise RuntimeError(f"unexpected case row in {path}: {case_id!r}")
        ids.append(case_id)
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate completed case row in {path}")
    return rows


def validate_resume_prefix(rows: list[dict], cases: list[dict], source: Path) -> None:
    observed = [row["case_id"] for row in rows]
    expected = [case["case_id"] for case in cases[: len(rows)]]
    if observed != expected:
        raise RuntimeError(
            f"{source}: completed rows are not the fixed case-order prefix"
        )


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(canonical_json(row) + "\n")
        output.flush()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_existing_metadata(path: Path, spec_sha256: str, current_head: str) -> dict | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if payload.get("runner_spec_sha256") != spec_sha256:
        raise RuntimeError("existing output metadata belongs to another runner spec")
    if payload.get("git_head") != current_head:
        raise RuntimeError("existing output metadata belongs to another git commit")
    return payload
