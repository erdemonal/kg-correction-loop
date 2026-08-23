import argparse
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SH, XSD

from src.controlled_cases import (
    CASE_BASE,
    KCL,
    ControlledCase,
    Statement,
    TypeAssertion,
    case_from_baseline_row,
    case_to_graph,
    class_uri,
    entity_uri,
    inject_addition,
    inject_cardinality_omission,
    inject_temporal_swap,
    relation_uri,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "experiments" / "controlled_selection.json"
OUTPUT_ROOT = ROOT / "outputs" / "controlled"
MANIFEST = OUTPUT_ROOT / "manifest.jsonl"

SOURCE_ROWS = {
    "movie": {
        "baseline": ROOT / "outputs" / "baseline" / "movie_llama31.jsonl",
        "stats": (
            ROOT
            / "outputs"
            / "evaluation"
            / "llama31"
            / "ont_1_movie_llm_stats.jsonl"
        ),
    },
    "music": {
        "baseline": ROOT / "outputs" / "baseline" / "music_llama31.jsonl",
        "stats": (
            ROOT
            / "outputs"
            / "evaluation"
            / "llama31"
            / "ont_2_music_llm_stats.jsonl"
        ),
    },
}

TAXONOMY = {
    "disjointness": "disjointness_violation",
    "domain_range": "domain_range_violation",
    "cardinality": "cardinality_breach",
    "temporal": "temporal_impossibility",
    "grounding": "grounding_error",
}

EXPECTED_SYMBOLIC = {
    "disjointness": {
        "raw_shacl": False,
        "owlrl_shacl": False,
        "owl_consistent": False,
    },
    "domain_range": {
        "raw_shacl": False,
        "owlrl_shacl": True,
        "owl_consistent": True,
    },
    "cardinality": {
        "raw_shacl": False,
        "owlrl_shacl": False,
        "owl_consistent": True,
    },
    "temporal": {
        "raw_shacl": False,
        "owlrl_shacl": False,
        "owl_consistent": True,
    },
    "grounding": {
        "raw_shacl": True,
        "owlrl_shacl": True,
        "owl_consistent": True,
    },
}


def read_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

    return rows


def index_unique(rows, path):
    indexed = {}

    for row in rows:
        case_id = row.get("id")

        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"Missing or invalid id in {path}")

        if case_id in indexed:
            raise RuntimeError(f"Duplicate id in {path}: {case_id}")

        indexed[case_id] = row

    return indexed


def load_selection(path=SELECTION):
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    cases = payload.get("cases")

    if not isinstance(cases, list):
        raise RuntimeError("controlled_selection.json has no cases list")

    ids = [row.get("id") for row in cases]

    if len(cases) != 50:
        raise RuntimeError(
            f"Expected 50 selected cases, found {len(cases)}"
        )

    if len(set(ids)) != 50:
        raise RuntimeError("Controlled selection contains duplicate ids")

    return payload


def load_canonical_rows():
    output = {}

    for domain, paths in SOURCE_ROWS.items():
        baseline = index_unique(
            read_jsonl(paths["baseline"]),
            paths["baseline"],
        )
        stats = index_unique(
            read_jsonl(paths["stats"]),
            paths["stats"],
        )

        joined = {}

        for case_id, generation in baseline.items():
            stat = stats.get(case_id)

            if stat is None:
                continue

            sent = stat.get("sent")
            triples = generation.get("triples")

            if not isinstance(sent, str) or not sent.strip():
                continue

            if not isinstance(triples, list):
                continue

            joined[case_id] = {
                "id": case_id,
                "domain": domain,
                "sent": sent,
                "triples": triples,
            }

        output[domain] = joined

    return output


def selected_row_source(selected, canonical_rows):
    domain = selected["domain"]
    case_id = selected["id"]

    row = canonical_rows[domain].get(case_id)

    if row is None:
        raise RuntimeError(
            f"Selected case is missing from local baseline/stats: {case_id}"
        )

    return row


def build_structural_case(selected, source_row):
    clean = case_from_baseline_row(source_row).with_derived_types()
    condition = selected["error_type"]
    proposal = selected["proposal"]

    if condition == "cardinality":
        subject = proposal["subject"]
        predicate = proposal["property"]
        value = proposal["value"]

        expected = Statement(subject, predicate, value)

        if expected not in clean.content:
            raise RuntimeError(
                f"{clean.case_id}: selected cardinality statement "
                "is not present in the clean graph"
            )

        injected = inject_cardinality_omission(
            clean,
            subject,
            predicate,
        )

    elif condition in {"disjointness", "grounding"}:
        subject, predicate, obj = proposal["injection"]
        injected = inject_addition(
            clean,
            Statement(subject, predicate, obj),
            TAXONOMY[condition],
        )

    else:
        raise RuntimeError(
            f"{clean.case_id}: unsupported structural condition {condition}"
        )

    return clean, injected


def build_domain_range_case(selected, source_row):
    domain = selected["domain"]
    proposal = selected["proposal"]
    subject, predicate, obj = proposal["statement"]
    types = []

    if domain == "movie":
        types.append(
            TypeAssertion(
                subject,
                "Q11424",
                "controlled:film_subject",
            )
        )
    else:
        types.append(
            TypeAssertion(
                subject,
                "Q134556",
                "controlled:single_subject",
            )
        )

    type_entity, type_class = proposal["explicit_type"]
    types.append(
        TypeAssertion(
            type_entity,
            type_class,
            "controlled:source_type",
        )
    )

    clean = ControlledCase(
        case_id=selected["id"],
        domain=domain,
        source_text=source_row["sent"],
        content=(),
        background_types=tuple(types),
    )

    injected = inject_addition(
        clean,
        Statement(subject, predicate, obj),
        TAXONOMY["domain_range"],
    )

    return clean, injected


def build_temporal_case(selected, source_row):
    domain = selected["domain"]
    proposal = selected["proposal"]
    subject = proposal["subject"]

    clean = ControlledCase(
        case_id=selected["id"],
        domain=domain,
        source_text=source_row["sent"],
        content=(
            Statement(
                subject,
                proposal["first_property"],
                proposal["first_value"],
                "date",
            ),
            Statement(
                subject,
                proposal["second_property"],
                proposal["second_value"],
                "date",
            ),
        ),
        background_types=(
            TypeAssertion(
                subject,
                "Q11424" if domain == "movie" else "Q2188189",
                "controlled:temporal_subject",
            ),
        ),
    )

    injected = inject_temporal_swap(
        clean,
        subject,
        proposal["first_property"],
        proposal["second_property"],
    )

    return clean, injected


def build_case_pair(selected, canonical_rows):
    source_row = selected_row_source(selected, canonical_rows)
    condition = selected["error_type"]

    if condition in {"cardinality", "disjointness", "grounding"}:
        return build_structural_case(selected, source_row)

    if condition == "domain_range":
        return build_domain_range_case(selected, source_row)

    if condition == "temporal":
        return build_temporal_case(selected, source_row)

    raise RuntimeError(
        f"{selected['id']}: unsupported condition {condition}"
    )


def _shape_uri(case_id, suffix):
    encoded = quote(case_id, safe="")
    return URIRef(
        f"{CASE_BASE}{encoded}/shape/{suffix}"
    )


def cardinality_shape(case_id, focus, property_uri):
    graph = Graph()
    shape = _shape_uri(case_id, "minimum-cardinality")
    property_shape = _shape_uri(
        case_id,
        "minimum-cardinality-property",
    )

    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, SH.targetNode, focus))
    graph.add((shape, SH.property, property_shape))
    graph.add((property_shape, SH.path, property_uri))
    graph.add((property_shape, SH.minCount, Literal(1)))
    graph.add(
        (
            property_shape,
            SH.message,
            Literal("The selected property must have at least one value."),
        )
    )

    return graph


def temporal_order_shape(
    case_id,
    focus,
    first_property,
    second_property,
):
    graph = Graph()
    shape = _shape_uri(case_id, "temporal-order")
    constraint = _shape_uri(case_id, "temporal-order-constraint")

    first_uri = KCL[first_property]
    second_uri = KCL[second_property]

    query = f"""
        SELECT $this
        WHERE {{
            $this <{first_uri}> ?first ;
                  <{second_uri}> ?second .
            FILTER (?first > ?second)
        }}
    """

    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, SH.targetNode, focus))
    graph.add((shape, SH.sparql, constraint))
    graph.add((constraint, RDF.type, SH.SPARQLConstraint))
    graph.add(
        (
            constraint,
            SH.message,
            Literal(
                "The first controlled event date must not be "
                "after the second controlled event date."
            ),
        )
    )
    graph.add((constraint, SH.select, Literal(query)))

    return graph


def temporal_owl_context(domain, properties):
    graph = Graph()
    domain_class = class_uri(
        domain,
        "Q11424" if domain == "movie" else "Q2188189",
    )

    for name in properties:
        property_uri = KCL[name]
        graph.add((property_uri, RDF.type, OWL.DatatypeProperty))
        graph.add((property_uri, RDFS.domain, domain_class))
        graph.add((property_uri, RDFS.range, XSD.date))

    return graph


def add_minimum_restriction(
    graph,
    case_id,
    focus,
    property_uri,
):
    encoded = quote(case_id, safe="")
    restriction = URIRef(
        f"{CASE_BASE}{encoded}/restriction/minimum-cardinality"
    )

    graph.add((focus, RDF.type, restriction))
    graph.add((restriction, RDF.type, OWL.Restriction))
    graph.add((restriction, OWL.onProperty, property_uri))
    graph.add(
        (
            restriction,
            OWL.minCardinality,
            Literal(1, datatype=XSD.nonNegativeInteger),
        )
    )


def build_symbolic_artifacts(selected, clean, injected):
    clean_graph = case_to_graph(clean)
    injected_graph = case_to_graph(injected)
    shapes = Graph()
    owl_context = Graph()
    condition = selected["error_type"]
    proposal = selected["proposal"]

    if condition == "cardinality":
        focus = entity_uri(
            selected["id"],
            proposal["subject"],
        )
        property_uri = relation_uri(
            selected["domain"],
            proposal["property"],
        )

        shapes = cardinality_shape(
            selected["id"],
            focus,
            property_uri,
        )

        add_minimum_restriction(
            clean_graph,
            selected["id"],
            focus,
            property_uri,
        )
        add_minimum_restriction(
            injected_graph,
            selected["id"],
            focus,
            property_uri,
        )

    elif condition == "temporal":
        focus = entity_uri(
            selected["id"],
            proposal["subject"],
        )

        shapes = temporal_order_shape(
            selected["id"],
            focus,
            proposal["first_property"],
            proposal["second_property"],
        )

        owl_context = temporal_owl_context(
            selected["domain"],
            (
                proposal["first_property"],
                proposal["second_property"],
            ),
        )

    return clean_graph, injected_graph, shapes, owl_context


def graph_to_file(graph, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(
        destination=str(path),
        format="turtle",
    )


def json_to_file(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def relative(path):
    return str(path.relative_to(ROOT))


def manifest_row(
    selected,
    clean,
    injected,
    case_dir,
):
    return {
        "id": selected["id"],
        "domain": selected["domain"],
        "condition": selected["error_type"],
        "taxonomy_error": TAXONOMY[selected["error_type"]],
        "source_text": clean.source_text,
        "primary_modification": (
            asdict(injected.primary_modification)
            if injected.primary_modification is not None
            else None
        ),
        "files": {
            "clean_graph": relative(case_dir / "clean.ttl"),
            "injected_graph": relative(case_dir / "injected.ttl"),
            "case_shapes": relative(case_dir / "case_shapes.ttl"),
            "owl_context": relative(case_dir / "owl_context.ttl"),
            "grounding_clean": relative(
                case_dir / "grounding_clean.json"
            ),
            "grounding_injected": relative(
                case_dir / "grounding_injected.json"
            ),
        },
        "expected_symbolic": {
            "clean": {
                "raw_shacl": True,
                "owlrl_shacl": True,
                "owl_consistent": True,
            },
            "injected": EXPECTED_SYMBOLIC[selected["error_type"]],
        },
    }


def build_all(selection=None, canonical_rows=None):
    if selection is None:
        selection = load_selection()

    if canonical_rows is None:
        canonical_rows = load_canonical_rows()

    bundles = []

    for selected in selection["cases"]:
        clean, injected = build_case_pair(
            selected,
            canonical_rows,
        )
        artifacts = build_symbolic_artifacts(
            selected,
            clean,
            injected,
        )
        bundles.append(
            {
                "selected": selected,
                "clean": clean,
                "injected": injected,
                "artifacts": artifacts,
            }
        )

    return bundles


def write_all(bundles, overwrite=False):
    if OUTPUT_ROOT.exists() and not overwrite:
        raise RuntimeError(
            f"{OUTPUT_ROOT} already exists. "
            "Use --overwrite to replace generated controlled outputs."
        )

    if OUTPUT_ROOT.exists():
        import shutil
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for bundle in bundles:
        selected = bundle["selected"]
        clean = bundle["clean"]
        injected = bundle["injected"]
        (
            clean_graph,
            injected_graph,
            shapes,
            owl_context,
        ) = bundle["artifacts"]

        case_dir = OUTPUT_ROOT / selected["id"]

        graph_to_file(clean_graph, case_dir / "clean.ttl")
        graph_to_file(
            injected_graph,
            case_dir / "injected.ttl",
        )
        graph_to_file(shapes, case_dir / "case_shapes.ttl")
        graph_to_file(
            owl_context,
            case_dir / "owl_context.ttl",
        )

        json_to_file(
            clean.grounding_payload(),
            case_dir / "grounding_clean.json",
        )
        json_to_file(
            injected.grounding_payload(),
            case_dir / "grounding_injected.json",
        )

        rows.append(
            manifest_row(
                selected,
                clean,
                injected,
                case_dir,
            )
        )

    with MANIFEST.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    return rows


def print_summary(rows):
    counts = {}

    for row in rows:
        key = (row["domain"], row["condition"])
        counts[key] = counts.get(key, 0) + 1

    for domain in ("movie", "music"):
        print(domain)

        for condition in (
            "disjointness",
            "domain_range",
            "cardinality",
            "temporal",
            "grounding",
        ):
            print(
                f"  {condition}: "
                f"{counts.get((domain, condition), 0)}"
            )

    print(f"total: {len(rows)}")
    print(f"wrote: {MANIFEST.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundles = build_all()
    rows = write_all(
        bundles,
        overwrite=args.overwrite,
    )
    print_summary(rows)


if __name__ == "__main__":
    main()
