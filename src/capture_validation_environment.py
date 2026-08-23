import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUTS = ROOT / "outputs" / "controlled"
RESULTS = ROOT / "results"

ENVIRONMENT_PATH = EXPERIMENTS / "environment.json"
MANIFEST_PATH = EXPERIMENTS / "validation_manifest.json"
ENRICHMENT_SPEC = EXPERIMENTS / "enrichment_spec.md"

PRIOR_HEAD = "f55d42e21bab1417fa6ab93bec00d8c668a2b035"

REQUIRED_SOURCE_FILES = [
    "experiments/error_taxonomy.json",
    "experiments/controlled_selection.json",
    "experiments/controlled_structural_selection.json",
    "experiments/controlled_targeted_selection.json",
    "experiments/controlled_review_decisions.jsonl",
    "experiments/enrichment_spec.md",
    "experiments/grounding_calibration_split.json",
    "experiments/grounding_judge_prompt_v1.txt",
    "experiments/grounding_judge_prompt_v2.txt",
    "experiments/grounding_judge_prompt_v3.txt",
    "experiments/grounding_judge_spec.json",
    "experiments/controlled_grounding_adjudication.json",
    "experiments/validation_protocol.md",
    "src/controlled_cases.py",
    "src/build_controlled_dataset.py",
    "src/validate_controlled_symbolic.py",
    "src/grounding_judge.py",
    "src/run_grounding_calibration.py",
    "src/run_controlled_grounding.py",
    "src/analyze_controlled_grounding.py",
    "validation/ontologies/movie_enrichment.ttl",
    "validation/ontologies/music_enrichment.ttl",
    "validation/shapes/movie_controlled.ttl",
    "validation/shapes/music_controlled.ttl",
]

REQUIRED_CORE_RESULTS = [
    "results/controlled_symbolic_validation.jsonl",
    "results/controlled_grounding_validation.jsonl",
    "results/controlled_grounding_validation.jsonl.meta.json",
    "results/controlled_grounding_target_analysis.json",
    "results/grounding_judge_v3_calibration.jsonl",
    "results/grounding_judge_v3_calibration.jsonl.meta.json",
    "results/grounding_judge_v3_heldout.jsonl",
    "results/grounding_judge_v3_heldout.jsonl.meta.json",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def run_command(args):
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return {
            "available": False,
            "error": str(exc),
        }

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def command_text(record):
    return "\n".join(
        part
        for part in (
            record.get("stdout", ""),
            record.get("stderr", ""),
        )
        if part
    )


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def replace_first_match(text, candidates, replacement, label):
    if replacement in text:
        return text

    for candidate in candidates:
        if candidate in text:
            return text.replace(candidate, replacement)

    raise RuntimeError(
        f"Could not find the expected {label} wording in enrichment_spec.md"
    )


def patch_enrichment_wording():
    text = ENRICHMENT_SPEC.read_text(encoding="utf-8")
    original = text

    text = replace_first_match(
        text,
        [
            (
                "SHACL is evaluated both on the original graph and, as a "
                "supplementary condition, after OWL RL materialization.\n\n"
                "Materialization may add the type implied by the OWL domain or "
                "range axiom. As a result, a SHACL violation in the original graph "
                "may disappear after materialization. The two conditions are "
                "reported separately."
            ),
            (
                "SHACL is evaluated both on the original graph and, as a "
                "supplementary condition, with the built in OWL RL inference "
                "option in pySHACL enabled and the ontology supplied to "
                "validation.\n\n"
                "This inference condition may add the type implied by an OWL domain "
                "or range axiom. As a result, a SHACL violation in the original "
                "graph may disappear when OWL RL inference is enabled. The two "
                "conditions are kept separate."
            ),
        ],
        (
            "SHACL is evaluated on the original graph and in a supplementary "
            "condition with pySHACL OWL RL inference enabled. The ontology is "
            "supplied during validation.\n\n"
            "OWL RL inference may add a type implied by an OWL domain or range "
            "axiom. A SHACL violation in the original graph may therefore "
            "disappear when this inference option is enabled. The two "
            "conditions are kept separate."
        ),
        "OWL RL",
    )

    text = replace_first_match(
        text,
        [
            (
                "The grounding assessor does not report an error because the "
                "statements that remain in the graph are still supported by the "
                "source text. The grounding task evaluates support for asserted "
                "statements, not extraction completeness."
            ),
            (
                "The controlled deletion itself cannot be detected by the grounding "
                "assessor because the grounding task evaluates support for asserted "
                "statements, not extraction completeness. Grounding findings on "
                "other baseline assertions, if any, remain background diagnostics."
            ),
        ],
        (
            "The grounding assessor cannot detect the controlled deletion itself "
            "because it checks support for assertions that are present. It does "
            "not check extraction completeness. Any grounding findings on other "
            "baseline assertions are kept as background results."
        ),
        "Movie cardinality",
    )

    text = replace_first_match(
        text,
        [
            (
                "The expected behavior is the same as in the Movie case: SHACL "
                "reports a violation, OWL remains consistent, and the grounding "
                "assessor does not report an unsupported statement."
            ),
            (
                "The expected symbolic behavior is the same as in the Movie case. "
                "SHACL reports a violation and OWL remains consistent. The "
                "controlled omission itself cannot be detected by the grounding "
                "assessor. Unrelated grounding findings remain background "
                "diagnostics."
            ),
        ],
        (
            "The expected symbolic behavior is the same as in the Movie case. "
            "SHACL reports a violation and OWL remains consistent. The grounding "
            "assessor cannot detect the omission itself. Any other grounding "
            "findings are kept as background results."
        ),
        "Music cardinality",
    )

    text = replace_first_match(
        text,
        [
            (
                "OWL remains consistent because the ontology does not contain "
                "axioms that compare the two date values."
            ),
            (
                "The intended OWL comparison contains no axiom that orders the two "
                "date values. In the current HermiT environment, xsd:date assertions "
                "are removed only from the HermiT compatibility view because of "
                "reasoner datatype support. The OWL arm therefore does not evaluate "
                "the temporal date assertions or their ordering."
            ),
        ],
        (
            "The OWL model contains no axiom that orders the two date values. "
            "In the current HermiT environment, `xsd:date` assertions are removed "
            "only from the copy sent to HermiT because the reasoner does not "
            "support that datatype in this setup. The OWL part of the study "
            "therefore does not examine the date assertions or their order."
        ),
        "temporal OWL",
    )

    if text != original:
        ENRICHMENT_SPEC.write_text(text, encoding="utf-8")
        return True

    return False


def hermit_jars():
    import owlready2

    package_root = Path(owlready2.__file__).resolve().parent
    jars = []

    for path in sorted(package_root.rglob("*.jar")):
        name_lower = path.name.lower()

        if "hermit" not in name_lower and "hermit" not in str(path.parent).lower():
            continue

        try:
            relative = str(path.relative_to(package_root))
        except ValueError:
            relative = path.name

        jars.append(
            {
                "relative_to_owlready2": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return jars


def git_record():
    head = run_command(["git", "rev-parse", "HEAD"])
    status = run_command(["git", "status", "--short"])
    git_version = run_command(["git", "--version"])

    return {
        "head": head.get("stdout"),
        "expected_prior_head": PRIOR_HEAD,
        "head_matches_expected_prior_commit": (
            head.get("stdout") == PRIOR_HEAD
        ),
        "status_short": status.get("stdout", ""),
        "git_version": command_text(git_version),
    }


def collect_environment():
    java = run_command(["java", "-version"])
    ollama = run_command(["ollama", "--version"])

    packages = {
        name: package_version(name)
        for name in (
            "rdflib",
            "pyshacl",
            "owlready2",
            "owlrl",
            "pytest",
            "pip",
        )
    }

    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Software environment used for controlled validation. "
            "Capturing this file does not run validators or the grounding model."
        ),
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "java": {
            "executable": shutil.which("java"),
            "version_output": command_text(java),
            "returncode": java.get("returncode"),
        },
        "ollama": {
            "executable": shutil.which("ollama"),
            "version_output": command_text(ollama),
            "returncode": ollama.get("returncode"),
        },
        "hermit_jars": hermit_jars(),
        "git": git_record(),
    }

    spec_path = EXPERIMENTS / "grounding_judge_spec.json"

    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        environment["grounding_assessor"] = {
            "judge_version": spec.get("judge_version"),
            "model": spec.get("model"),
            "model_digest": spec.get("model_digest"),
            "prompt_sha256": spec.get("prompt_sha256"),
        }

    return environment


def file_record(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_paths(relative_paths):
    missing = []

    for relative in relative_paths:
        if not (ROOT / relative).is_file():
            missing.append(relative)

    if missing:
        joined = "\n  ".join(missing)
        raise RuntimeError(f"Missing required files:\n  {joined}")


def controlled_output_records():
    if not OUTPUTS.is_dir():
        raise RuntimeError(
            "outputs/controlled is missing. Controlled graphs must exist "
            "before the validation manifest is captured."
        )

    manifest_path = OUTPUTS / "manifest.jsonl"

    if not manifest_path.is_file():
        raise RuntimeError("outputs/controlled/manifest.jsonl is missing")

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(rows) != 50:
        raise RuntimeError(
            f"Expected 50 controlled manifest rows, found {len(rows)}"
        )

    case_ids = [row.get("id") for row in rows]

    if len(set(case_ids)) != 50:
        raise RuntimeError("Controlled manifest does not contain 50 unique ids")

    records = [
        file_record(path)
        for path in sorted(OUTPUTS.rglob("*"))
        if path.is_file()
    ]

    return rows, records


def result_records():
    require_paths(REQUIRED_CORE_RESULTS)

    paths = set()

    for pattern in (
        "controlled*",
        "grounding_judge_v*",
    ):
        for path in RESULTS.glob(pattern):
            if path.is_file():
                paths.add(path)

    return [file_record(path) for path in sorted(paths)]


def source_records():
    require_paths(REQUIRED_SOURCE_FILES)

    extra_validation = [
        path
        for base in (
            ROOT / "validation" / "ontologies",
            ROOT / "validation" / "shapes",
        )
        for path in base.rglob("*")
        if path.is_file()
    ]

    paths = {
        ROOT / relative
        for relative in REQUIRED_SOURCE_FILES
    }
    paths.update(extra_validation)

    return [file_record(path) for path in sorted(paths)]


def selection_summary():
    path = EXPERIMENTS / "controlled_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("selection_size") != 50:
        raise RuntimeError("controlled_selection.json is not the frozen 50-case set")

    return {
        "selection_size": payload.get("selection_size"),
        "unique_case_ids": payload.get("unique_case_ids"),
        "counts": payload.get("counts"),
    }


def write_manifest(environment):
    controlled_rows, outputs = controlled_output_records()
    source = source_records()

    environment_record = file_record(ENVIRONMENT_PATH)
    results = result_records()

    manifest = {
        "version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Hashes for the controlled validation setup. Generated graphs "
            "and result files stay outside Git. Their hashes are recorded "
            "here before the repair experiments begin."
        ),
        "prior_repository_commit": PRIOR_HEAD,
        "selection": selection_summary(),
        "controlled_manifest_rows": len(controlled_rows),
        "convergence_rule": (
            "Converged when both the violation identity set and the asserted "
            "triple set are unchanged across two consecutive repair rounds."
        ),
        "source_artifacts": source,
        "environment_artifact": environment_record,
        "controlled_outputs": outputs,
        "results": results,
        "counts": {
            "source_artifacts": len(source),
            "controlled_output_files": len(outputs),
            "result_files": len(results),
        },
        "grounding_identity": {
            "judge_version": environment.get(
                "grounding_assessor", {}
            ).get("judge_version"),
            "model": environment.get(
                "grounding_assessor", {}
            ).get("model"),
            "model_digest": environment.get(
                "grounding_assessor", {}
            ).get("model_digest"),
            "prompt_sha256": environment.get(
                "grounding_assessor", {}
            ).get("prompt_sha256"),
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return manifest


def main():
    patched = patch_enrichment_wording()
    environment = collect_environment()

    if not environment["hermit_jars"]:
        raise RuntimeError(
            "No HermiT JAR was found inside the installed Owlready2 package."
        )

    if environment["packages"]["owlready2"] is None:
        raise RuntimeError("Owlready2 version could not be captured")

    if environment["packages"]["pyshacl"] is None:
        raise RuntimeError("pySHACL version could not be captured")

    if environment["packages"]["rdflib"] is None:
        raise RuntimeError("RDFLib version could not be captured")

    ENVIRONMENT_PATH.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = write_manifest(environment)

    print(
        "enrichment wording patched:"
        f" {'yes' if patched else 'already final'}"
    )
    print(f"python: {platform.python_version()}")
    print(
        "java: "
        + environment["java"]["version_output"].splitlines()[0]
    )
    print(
        "packages: "
        + ", ".join(
            f"{name}={version}"
            for name, version in environment["packages"].items()
            if version is not None
        )
    )
    print(
        "HermiT JARs: "
        f"{len(environment['hermit_jars'])}"
    )

    for jar in environment["hermit_jars"]:
        print(
            "  "
            f"{jar['relative_to_owlready2']} "
            f"sha256={jar['sha256']}"
        )

    print(
        "controlled cases: "
        f"{manifest['controlled_manifest_rows']}"
    )
    print(
        "controlled output files hashed: "
        f"{manifest['counts']['controlled_output_files']}"
    )
    print(
        "result files hashed: "
        f"{manifest['counts']['result_files']}"
    )
    print(
        "git HEAD before this record: "
        f"{environment['git']['head']}"
    )
    print(
        "expected prior HEAD matched: "
        f"{environment['git']['head_matches_expected_prior_commit']}"
    )
    print(
        f"wrote: {ENVIRONMENT_PATH.relative_to(ROOT)}"
    )
    print(
        f"wrote: {MANIFEST_PATH.relative_to(ROOT)}"
    )
    print("No validator or grounding model was executed.")


if __name__ == "__main__":
    main()
