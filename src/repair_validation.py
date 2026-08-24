import hashlib
import json

from pyshacl import validate
from rdflib.namespace import RDF, SH

from src.build_controlled_dataset import (
    add_minimum_restriction,
    build_case_pair,
    build_symbolic_artifacts,
    load_canonical_rows,
    load_selection,
)
from src.controlled_cases import (
    RELATIONS,
    ControlledCase,
    Statement,
    case_to_graph,
    entity_uri,
    relation_uri,
)
from src.validate_controlled_symbolic import (
    ontology_graph,
    owl_consistent,
    shapes_graph,
)


def canonical_term(term):
    return None if term is None else term.n3()


def shacl_violation_identity(result_graph, result_node):
    fields = {
        "sourceConstraintComponent": result_graph.value(result_node, SH.sourceConstraintComponent),
        "focusNode": result_graph.value(result_node, SH.focusNode),
        "resultPath": result_graph.value(result_node, SH.resultPath),
        "value": result_graph.value(result_node, SH.value),
        "sourceShape": result_graph.value(result_node, SH.sourceShape),
    }
    canonical = {key: canonical_term(value) for key, value in fields.items()}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return {"violation_id": f"shacl:{digest}", "identity": canonical}


def shacl_results(data_graph, shape_graph):
    conforms, report_graph, _ = validate(
        data_graph=data_graph,
        shacl_graph=shape_graph,
        inference="none",
    )
    results = []
    for node in report_graph.subjects(RDF.type, SH.ValidationResult):
        identity = shacl_violation_identity(report_graph, node)
        message = report_graph.value(node, SH.resultMessage)
        results.append({
            **identity,
            "message": None if message is None else str(message),
            "focus_node": canonical_term(report_graph.value(node, SH.focusNode)),
            "result_path": canonical_term(report_graph.value(node, SH.resultPath)),
            "value": canonical_term(report_graph.value(node, SH.value)),
        })
    results.sort(key=lambda row: row["violation_id"])
    return {"conforms": bool(conforms), "violations": results}


def selected_case(case_id):
    matches = [row for row in load_selection()["cases"] if row["id"] == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one selected case for {case_id}, found {len(matches)}")
    return matches[0]


def controlled_context(case_id):
    selected = selected_case(case_id)
    canonical_rows = load_canonical_rows()
    clean, injected = build_case_pair(selected, canonical_rows)
    _, _, case_shapes, owl_context = build_symbolic_artifacts(selected, clean, injected)
    return {
        "selected": selected,
        "clean": clean,
        "injected": injected,
        "case_shapes": case_shapes,
        "owl_context": owl_context,
    }


def temporal_relations(context):
    names = set()
    for case in (context["clean"], context["injected"]):
        for statement in case.content:
            if statement.object_kind == "date":
                names.add(statement.predicate)
    return names


def allowed_relations(context):
    names = set(RELATIONS[context["selected"]["domain"]])
    names.update(temporal_relations(context))
    return tuple(sorted(names))


def statements_from_triples(context, triples):
    allowed = set(allowed_relations(context))
    temporal = temporal_relations(context)
    statements = []
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3 or not all(isinstance(value, str) for value in triple):
            raise ValueError(f"Invalid repair triple: {triple!r}")
        subject, predicate, obj = triple
        if predicate not in allowed:
            raise ValueError(f"Relation is not allowed for this case: {predicate}")
        statements.append(Statement(subject, predicate, obj, "date" if predicate in temporal else "entity"))
    return tuple(statements)


def repaired_case(context, triples):
    clean = context["clean"]
    return ControlledCase(
        case_id=clean.case_id,
        domain=clean.domain,
        source_text=clean.source_text,
        content=statements_from_triples(context, triples),
        background_types=clean.background_types,
    )


def symbolic_graph(context, repair_case):
    selected = context["selected"]
    graph = case_to_graph(repair_case)
    if selected["error_type"] == "cardinality":
        proposal = selected["proposal"]
        focus = entity_uri(selected["id"], proposal["subject"])
        property_uri = relation_uri(selected["domain"], proposal["property"])
        add_minimum_restriction(graph, selected["id"], focus, property_uri)
    return graph


def validation_context_graphs(context):
    domain = context["selected"]["domain"]
    ontology = ontology_graph(domain, context["owl_context"])
    shapes = shapes_graph(domain, context["case_shapes"])
    return shapes, ontology


def grounding_payload(repair_case):
    return repair_case.grounding_payload()


def revalidate_symbolic(context, triples):
    repair_case = repaired_case(context, triples)
    data_graph = symbolic_graph(context, repair_case)
    shapes, ontology = validation_context_graphs(context)
    return {
        "shacl": shacl_results(data_graph, shapes),
        "owl_consistent": owl_consistent(data_graph, ontology),
        "grounding_payload": grounding_payload(repair_case),
    }


def controlled_owl_focus(context, triples):
    selected = context["selected"]
    if selected["error_type"] != "disjointness":
        return None
    details = context["injected"].primary_modification.details
    target = (details["subject"], details["predicate"], details["object"])
    if target not in {tuple(triple) for triple in triples}:
        return None
    if selected["domain"] == "movie":
        return details["object"]
    if selected["domain"] == "music":
        return details["subject"]
    return None


def owl_feedback(context, triples, is_consistent):
    if is_consistent:
        return None
    case_id = context["selected"]["id"]
    focus = controlled_owl_focus(context, triples)
    return {
        "validator": "owl_consistency",
        "violation_id": f"owl:inconsistent:{case_id}",
        "error_type": "disjointness_violation" if focus is not None else None,
        "focus": focus,
        "message": "The graph is logically inconsistent.",
    }
