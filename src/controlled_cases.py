from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal
from urllib.parse import quote

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, SH, XSD


KCL = Namespace("https://github.com/erdemonal/kg-correction-loop#")
CASE_BASE = "https://github.com/erdemonal/kg-correction-loop#case/"

MOVIE_CONCEPTS = Namespace(
    "https://cenguix.github.io/Text2KGBench/ont_1_movie/concepts#"
)
MOVIE_RELATIONS = Namespace(
    "https://cenguix.github.io/Text2KGBench/ont_1_movie/relations#"
)
MUSIC_CONCEPTS = Namespace(
    "https://cenguix.github.io/Text2KGBench/ont_2_music/concepts#"
)
MUSIC_RELATIONS = Namespace(
    "https://cenguix.github.io/Text2KGBench/ont_2_music/relations#"
)

RELATIONS = {
    "movie": {
        "genre": "P136",
        "nominated_for": "P1411",
        "based_on": "P144",
        "cast_member": "P161",
        "award_received": "P166",
        "cost": "P2130",
        "production_company": "P272",
        "country_of_origin": "P495",
        "director": "P57",
        "publication_date": "P577",
        "screenwriter": "P58",
        "characters": "P674",
        "narrative_location": "P840",
        "filming_location": "P915",
        "main_subject": "P921",
    },
    "music": {
        "occupation": "P106",
        "genre": "P136",
        "nominated_for": "P1411",
        "producer": "P162",
        "performer": "P175",
        "record_label": "P264",
        "part_of": "P361",
        "language": "P407",
        "language_of_work_or_name": "P407",
        "voice_type": "P412",
        "publication_date": "P577",
        "tracklist": "P658",
        "lyrics_by": "P676",
        "composer": "P86",
        "instrumentation": "P870",
    },
}

DOMAIN_TYPES = {
    "movie": {
        "genre": "Q11424",
        "nominated_for": "Q11424",
        "based_on": "Q11424",
        "cast_member": "Q11424",
        "award_received": "Q11424",
        "cost": "Q11424",
        "production_company": "Q11424",
        "country_of_origin": "Q11424",
        "director": "Q11424",
        "publication_date": "Q11424",
        "screenwriter": "Q11424",
        "characters": "Q11424",
        "narrative_location": "Q11424",
        "filming_location": "Q11424",
        "main_subject": "Q11424",
    },
    "music": {
        "occupation": "Q5",
        "genre": "Q2188189",
        "nominated_for": "Q482994",
        "producer": "Q482994",
        "performer": "Q2188189",
        "record_label": "Q482994",
        "part_of": "Q2188189",
        "language": "Q2188189",
        "language_of_work_or_name": "Q2188189",
        "voice_type": "Q5",
        "publication_date": "Q2188189",
        "tracklist": "Q482994",
        "lyrics_by": "Q2188189",
        "composer": "Q2188189",
        "instrumentation": "Q2188189",
    },
}

RANGE_TYPES = {
    "movie": {
        "genre": "Q483394",
        "nominated_for": "Q618779",
        "based_on": "Q47461344",
        "cast_member": "Q5",
        "award_received": "Q618779",
        "production_company": "Q1762059",
        "country_of_origin": "Q6256",
        "director": "Q5",
        "screenwriter": "Q5",
        "characters": "Q15773347",
        "narrative_location": "Q515",
        "filming_location": "Q515",
    },
    "music": {
        "occupation": "Q66715801",
        "genre": "Q188451",
        "nominated_for": "Q618779",
        "producer": "Q5",
        "performer": "Q5",
        "part_of": "Q482994",
        "language": "Q34770",
        "language_of_work_or_name": "Q34770",
        "voice_type": "Q7390",
        "tracklist": "Q2188189",
        "lyrics_by": "Q5",
        "composer": "Q5",
    },
}


@dataclass(frozen=True)
class Statement:
    subject: str
    predicate: str
    object: str
    object_kind: Literal["entity", "date"] = "entity"


@dataclass(frozen=True)
class TypeAssertion:
    entity: str
    class_id: str
    provenance: str


@dataclass(frozen=True)
class PrimaryModification:
    error_type: str
    operation: str
    details: dict


@dataclass(frozen=True)
class ControlledCase:
    case_id: str
    domain: Literal["movie", "music"]
    source_text: str
    content: tuple[Statement, ...]
    background_types: tuple[TypeAssertion, ...] = ()
    primary_modification: PrimaryModification | None = None

    def with_derived_types(self):
        known = {
            (item.entity, item.class_id, item.provenance)
            for item in self.background_types
        }

        additions = []

        for statement in self.content:
            if statement.object_kind != "entity":
                continue

            domain_type = DOMAIN_TYPES[self.domain].get(statement.predicate)
            range_type = RANGE_TYPES[self.domain].get(statement.predicate)

            if domain_type is not None:
                value = (
                    statement.subject,
                    domain_type,
                    f"domain:{statement.predicate}",
                )
                if value not in known:
                    additions.append(TypeAssertion(*value))
                    known.add(value)

            if range_type is not None:
                value = (
                    statement.object,
                    range_type,
                    f"range:{statement.predicate}",
                )
                if value not in known:
                    additions.append(TypeAssertion(*value))
                    known.add(value)

        return replace(
            self,
            background_types=self.background_types + tuple(additions),
        )

    def grounding_payload(self):
        return {
            "id": self.case_id,
            "domain": self.domain,
            "source_text": self.source_text,
            "triples": [
                [s.subject, s.predicate, s.object]
                for s in self.content
            ],
        }


def case_from_baseline_row(row):
    case_id = row.get("id")
    domain = row.get("domain")
    source_text = row.get("sent")
    triples = row.get("triples")

    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Missing case id")

    if domain not in RELATIONS:
        raise ValueError(f"Unsupported domain: {domain!r}")

    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError(f"{case_id}: missing source text")

    if not isinstance(triples, list):
        raise ValueError(f"{case_id}: triples must be a list")

    content = []

    for triple in triples:
        if (
            not isinstance(triple, list)
            or len(triple) != 3
            or not all(isinstance(value, str) for value in triple)
        ):
            raise ValueError(f"{case_id}: invalid triple: {triple!r}")

        subject, predicate, obj = triple

        if predicate not in RELATIONS[domain]:
            raise ValueError(
                f"{case_id}: relation is not in the pinned {domain} ontology: "
                f"{predicate}"
            )

        content.append(Statement(subject, predicate, obj))

    return ControlledCase(
        case_id=case_id,
        domain=domain,
        source_text=source_text,
        content=tuple(content),
    )


def add_background_type(case, entity, class_id, provenance):
    assertion = TypeAssertion(entity, class_id, provenance)

    if assertion in case.background_types:
        return case

    return replace(
        case,
        background_types=case.background_types + (assertion,),
    )


def relation_uri(domain, predicate):
    relation_id = RELATIONS[domain].get(predicate)

    if relation_id is None:
        raise ValueError(
            f"Relation is not in the pinned {domain} ontology: {predicate}"
        )

    namespace = MOVIE_RELATIONS if domain == "movie" else MUSIC_RELATIONS
    return namespace[relation_id]


def class_uri(domain, class_id):
    namespace = MOVIE_CONCEPTS if domain == "movie" else MUSIC_CONCEPTS
    return namespace[class_id]


def entity_uri(case_id, label):
    digest = sha256(label.encode("utf-8")).hexdigest()[:12]
    encoded_case = quote(case_id, safe="")
    encoded_label = quote(label.strip(), safe="")[:96]
    return URIRef(f"{CASE_BASE}{encoded_case}/{encoded_label}-{digest}")


def case_to_graph(case):
    graph = Graph()

    for statement in case.content:
        subject = entity_uri(case.case_id, statement.subject)
        predicate = (
            KCL[statement.predicate]
            if statement.object_kind == "date"
            else relation_uri(case.domain, statement.predicate)
        )

        if statement.object_kind == "date":
            obj = Literal(statement.object, datatype=XSD.date)
        else:
            obj = entity_uri(case.case_id, statement.object)

        graph.add((subject, predicate, obj))

    for assertion in case.background_types:
        graph.add(
            (
                entity_uri(case.case_id, assertion.entity),
                RDF.type,
                class_uri(case.domain, assertion.class_id),
            )
        )

    return graph


def add_min_cardinality_restriction(graph, focus_node, property_uri):
    restriction = BNode()
    graph.add((focus_node, RDF.type, restriction))
    graph.add((restriction, RDF.type, OWL.Restriction))
    graph.add((restriction, OWL.onProperty, property_uri))
    graph.add(
        (
            restriction,
            OWL.minCardinality,
            Literal(1, datatype=XSD.nonNegativeInteger),
        )
    )
    return restriction


def min_count_shape(focus_node, property_uri):
    graph = Graph()
    shape = BNode()
    property_shape = BNode()

    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, SH.targetNode, focus_node))
    graph.add((shape, SH.property, property_shape))
    graph.add((property_shape, SH.path, property_uri))
    graph.add((property_shape, SH.minCount, Literal(1)))

    return graph


def _ensure_unmodified(case):
    if case.primary_modification is not None:
        raise ValueError(
            f"{case.case_id}: a primary modification has already been applied"
        )


def inject_addition(case, statement, error_type):
    _ensure_unmodified(case)

    if statement in case.content:
        raise ValueError(f"{case.case_id}: statement already exists")

    return replace(
        case,
        content=case.content + (statement,),
        primary_modification=PrimaryModification(
            error_type=error_type,
            operation="add",
            details={
                "subject": statement.subject,
                "predicate": statement.predicate,
                "object": statement.object,
                "object_kind": statement.object_kind,
            },
        ),
    )


def inject_cardinality_omission(case, subject, predicate):
    _ensure_unmodified(case)

    matches = [
        statement
        for statement in case.content
        if statement.subject == subject
        and statement.predicate == predicate
    ]

    if len(matches) != 1:
        raise ValueError(
            f"{case.case_id}: expected exactly one {predicate} statement "
            f"for {subject!r}, found {len(matches)}"
        )

    removed = matches[0]
    content = tuple(
        statement
        for statement in case.content
        if statement != removed
    )

    return replace(
        case,
        content=content,
        primary_modification=PrimaryModification(
            error_type="cardinality_breach",
            operation="remove",
            details={
                "subject": removed.subject,
                "predicate": removed.predicate,
                "object": removed.object,
            },
        ),
    )


def inject_temporal_swap(case, subject, first_predicate, second_predicate):
    _ensure_unmodified(case)

    first = [
        statement
        for statement in case.content
        if statement.subject == subject
        and statement.predicate == first_predicate
        and statement.object_kind == "date"
    ]
    second = [
        statement
        for statement in case.content
        if statement.subject == subject
        and statement.predicate == second_predicate
        and statement.object_kind == "date"
    ]

    if len(first) != 1 or len(second) != 1:
        raise ValueError(
            f"{case.case_id}: temporal swap requires one value for each "
            "temporal predicate"
        )

    first_statement = first[0]
    second_statement = second[0]

    swapped = []

    for statement in case.content:
        if statement == first_statement:
            swapped.append(
                Statement(
                    subject,
                    first_predicate,
                    second_statement.object,
                    "date",
                )
            )
        elif statement == second_statement:
            swapped.append(
                Statement(
                    subject,
                    second_predicate,
                    first_statement.object,
                    "date",
                )
            )
        else:
            swapped.append(statement)

    return replace(
        case,
        content=tuple(swapped),
        primary_modification=PrimaryModification(
            error_type="temporal_impossibility",
            operation="swap",
            details={
                "subject": subject,
                "first_predicate": first_predicate,
                "second_predicate": second_predicate,
                "first_value": first_statement.object,
                "second_value": second_statement.object,
            },
        ),
    )
