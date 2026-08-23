import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from owlready2 import (
    OwlReadyInconsistentOntologyError,
    World,
    sync_reasoner,
)
from pyshacl import validate
from rdflib import Graph, Literal
from rdflib.namespace import XSD


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "controlled" / "manifest.jsonl"
RESULTS = (
    ROOT
    / "results"
    / "controlled_symbolic_validation.jsonl"
)

PINNED_COMMIT = "50a3d255371b8817cdff70fd88459ac82b339cfe"

SOURCE_ONTOLOGIES = {
    "movie": {
        "path": ROOT / "validation" / "source" / "ont_1_movie.ttl",
        "url": (
            "https://raw.githubusercontent.com/cenguix/Text2KGBench/"
            f"{PINNED_COMMIT}/data/wikidata_tekgen/ontologies/owl/"
            "ont_1_movie.ttl"
        ),
        "git_blob_sha1": "b7b43c30d7dd2df8bc409c8b5202b1fc80a49ef0",
        "enrichment": (
            ROOT
            / "validation"
            / "ontologies"
            / "movie_enrichment.ttl"
        ),
        "shapes": (
            ROOT
            / "validation"
            / "shapes"
            / "movie_controlled.ttl"
        ),
    },
    "music": {
        "path": ROOT / "validation" / "source" / "ont_2_music.ttl",
        "url": (
            "https://raw.githubusercontent.com/cenguix/Text2KGBench/"
            f"{PINNED_COMMIT}/data/wikidata_tekgen/ontologies/owl/"
            "ont_2_music.ttl"
        ),
        "git_blob_sha1": "a1adc56eb0b92ecbd5b5ff5f44a67a59d0447539",
        "enrichment": (
            ROOT
            / "validation"
            / "ontologies"
            / "music_enrichment.ttl"
        ),
        "shapes": (
            ROOT
            / "validation"
            / "shapes"
            / "music_controlled.ttl"
        ),
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


def git_blob_sha1(data):
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def ensure_source_ontology(domain):
    config = SOURCE_ONTOLOGIES[domain]
    path = config["path"]

    if path.exists():
        data = path.read_bytes()

        if git_blob_sha1(data) != config["git_blob_sha1"]:
            raise RuntimeError(
                f"Cached source ontology does not match the pinned blob: "
                f"{path}"
            )

        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(config["url"], timeout=30) as response:
        data = response.read()

    actual = git_blob_sha1(data)

    if actual != config["git_blob_sha1"]:
        raise RuntimeError(
            f"Downloaded {domain} ontology does not match the pinned blob. "
            f"Expected {config['git_blob_sha1']}, got {actual}."
        )

    path.write_bytes(data)
    return path


def load_graph(path):
    graph = Graph()

    if path.stat().st_size:
        graph.parse(path, format="turtle")

    return graph


def merged_graph(*graphs):
    output = Graph()

    for graph in graphs:
        for triple in graph:
            output.add(triple)

    return output


def ontology_graph(domain, owl_context):
    config = SOURCE_ONTOLOGIES[domain]
    source = load_graph(ensure_source_ontology(domain))
    enrichment = load_graph(config["enrichment"])

    return merged_graph(
        source,
        enrichment,
        owl_context,
    )


def shapes_graph(domain, case_shapes):
    config = SOURCE_ONTOLOGIES[domain]
    core = load_graph(config["shapes"])

    return merged_graph(core, case_shapes)


def shacl_conforms(
    data_graph,
    shapes,
    *,
    inference="none",
    ont_graph=None,
):
    conforms, _, _ = validate(
        data_graph=data_graph,
        shacl_graph=shapes,
        ont_graph=ont_graph,
        inference=inference,
    )
    return bool(conforms)


def hermit_compatible_graph(graph):
    output = Graph()

    for subject, predicate, obj in graph:
        if obj == XSD.date:
            continue

        if (
            isinstance(obj, Literal)
            and obj.datatype == XSD.date
        ):
            continue

        output.add((subject, predicate, obj))

    return output


def owl_consistent(data_graph, ontology):
    graph = hermit_compatible_graph(
        merged_graph(ontology, data_graph)
    )

    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "controlled.rdf"
        graph.serialize(
            destination=str(path),
            format="xml",
        )

        world = World()
        onto = world.get_ontology(path.as_uri()).load()

        try:
            sync_reasoner([onto], debug=0)
        except OwlReadyInconsistentOntologyError:
            return False

    return True


def validate_variant(
    domain,
    data_graph,
    shapes,
    ontology,
):
    return {
        "raw_shacl": shacl_conforms(
            data_graph,
            shapes,
            inference="none",
        ),
        "owlrl_shacl": shacl_conforms(
            data_graph,
            shapes,
            inference="owlrl",
            ont_graph=ontology,
        ),
        "owl_consistent": owl_consistent(
            data_graph,
            ontology,
        ),
    }


def validate_case(row):
    files = {
        name: ROOT / relative
        for name, relative in row["files"].items()
    }

    clean = load_graph(files["clean_graph"])
    injected = load_graph(files["injected_graph"])
    case_shapes = load_graph(files["case_shapes"])
    owl_context = load_graph(files["owl_context"])

    ontology = ontology_graph(
        row["domain"],
        owl_context,
    )
    shapes = shapes_graph(
        row["domain"],
        case_shapes,
    )

    observed = {
        "clean": validate_variant(
            row["domain"],
            clean,
            shapes,
            ontology,
        ),
        "injected": validate_variant(
            row["domain"],
            injected,
            shapes,
            ontology,
        ),
    }

    mismatches = []

    for variant in ("clean", "injected"):
        expected = row["expected_symbolic"][variant]

        for validator_name, expected_value in expected.items():
            observed_value = observed[variant][validator_name]

            if observed_value != expected_value:
                mismatches.append(
                    {
                        "variant": variant,
                        "validator": validator_name,
                        "expected": expected_value,
                        "observed": observed_value,
                    }
                )

    return {
        "id": row["id"],
        "domain": row["domain"],
        "condition": row["condition"],
        "observed": observed,
        "mismatches": mismatches,
    }


def write_results(results):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def print_summary(results):
    mismatch_cases = [
        row
        for row in results
        if row["mismatches"]
    ]

    print(f"cases: {len(results)}")
    print(f"matching expected symbolic pattern: "
          f"{len(results) - len(mismatch_cases)}")
    print(f"mismatch cases: {len(mismatch_cases)}")

    counts = {}

    for row in results:
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

    if mismatch_cases:
        print("mismatches:")

        for row in mismatch_cases:
            print(f"  {row['id']}")

            for mismatch in row["mismatches"]:
                print(
                    "    "
                    f"{mismatch['variant']} / "
                    f"{mismatch['validator']}: "
                    f"expected {mismatch['expected']}, "
                    f"observed {mismatch['observed']}"
                )

    print(f"wrote: {RESULTS.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(
            "Controlled manifest not found. Run "
            "`python -m src.build_controlled_dataset` first."
        )

    rows = read_jsonl(MANIFEST)
    results = []

    for index, row in enumerate(rows, start=1):
        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{row['id']} "
            f"({row['condition']})"
        )
        results.append(validate_case(row))

    write_results(results)
    print_summary(results)

    mismatches = sum(
        bool(row["mismatches"])
        for row in results
    )

    if mismatches and not args.allow_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
