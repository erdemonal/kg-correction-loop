import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src import extract_text2kg
from src import grounding_judge
from src.repair_engine import (
    normalize_triples,
    primary_delta,
    reference_metrics,
    stop_reason_after_repair,
    trajectory_state,
    triple_set,
)
from src.repair_validation import (
    allowed_relations,
    controlled_context,
    owl_feedback,
    repaired_case,
    shacl_results,
    symbolic_graph,
    revalidate_symbolic,
)


ROOT = Path(__file__).resolve().parents[1]

SPEC_PATH = ROOT / "experiments" / "repair_spec.json"
PROMPT_PATH = ROOT / "experiments" / "repair_prompt.txt"
BASELINE_PATH = (
    ROOT / "experiments" / "text2kgbench_llama31_baseline.json"
)
GROUNDING_SPEC_PATH = (
    ROOT / "experiments" / "grounding_judge_spec.json"
)
FROZEN_GROUNDING_RESULTS = (
    ROOT / "results" / "controlled_grounding_validation.jsonl"
)
FROZEN_TARGET_ANALYSIS = (
    ROOT / "results" / "controlled_grounding_target_analysis.json"
)

DEFAULT_OUTPUT = (
    ROOT / "results" / "controlled_repair_trajectories.jsonl"
)

REPAIR_MODEL = "llama3.1:8b-instruct-q4_K_M"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def index_unique(rows, source):
    output = {}

    for row in rows:
        case_id = row.get("id")

        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"{source}: missing case id")

        if case_id in output:
            raise RuntimeError(
                f"{source}: duplicate case id {case_id}"
            )

        output[case_id] = row

    return output


def git_head():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def load_spec():
    spec = read_json(SPEC_PATH)

    prompt = PROMPT_PATH.read_bytes()
    actual_prompt_sha = sha256_bytes(prompt)
    expected_prompt_sha = spec["prompt"]["sha256"]

    if actual_prompt_sha != expected_prompt_sha:
        raise RuntimeError(
            "Repair prompt hash does not match repair_spec.json. "
            f"Expected {expected_prompt_sha}, got {actual_prompt_sha}."
        )

    baseline = read_json(BASELINE_PATH)
    model_spec = spec["repair_model"]
    baseline_model = baseline["model"]

    if model_spec["name"] != baseline_model["name"]:
        raise RuntimeError(
            "Repair model name does not match the extraction baseline"
        )

    expected_options = {
        "temperature": baseline_model["temperature"],
        "seed": baseline_model["seed"],
        "num_ctx": baseline_model["num_ctx"],
        "num_predict": baseline_model["num_predict"],
    }

    if model_spec["options"] != expected_options:
        raise RuntimeError(
            "Repair generation settings do not match the "
            "extraction baseline"
        )

    return spec, baseline


def verify_grounding_spec():
    spec = read_json(GROUNDING_SPEC_PATH)

    if spec.get("status") != "frozen":
        raise RuntimeError("Grounding judge is not marked frozen")

    if spec.get("judge_version") != grounding_judge.JUDGE_VERSION:
        raise RuntimeError(
            "Grounding judge version differs from frozen spec"
        )

    if spec.get("model") != grounding_judge.MODEL:
        raise RuntimeError(
            "Grounding judge model differs from frozen spec"
        )

    if spec.get("model_digest") != grounding_judge.EXPECTED_DIGEST:
        raise RuntimeError(
            "Grounding judge digest differs from frozen spec"
        )

    prompt_path = ROOT / spec["prompt"]

    if sha256_file(prompt_path) != spec["prompt_sha256"]:
        raise RuntimeError(
            "Grounding judge prompt differs from frozen spec"
        )

    if spec["options"] != grounding_judge.OPTIONS:
        raise RuntimeError(
            "Grounding judge options differ from frozen spec"
        )

    return spec


def local_repair_model_metadata(spec, baseline):
    version = extract_text2kg.api("/api/version")["version"]
    models = extract_text2kg.api("/api/tags")["models"]

    model_name = spec["repair_model"]["name"]
    model = next(
        (
            row
            for row in models
            if row.get("name") == model_name
            or row.get("model") == model_name
        ),
        None,
    )

    if model is None:
        raise RuntimeError(
            f"Repair model not found in Ollama: {model_name}"
        )

    digest = model.get("digest")
    expected_digest = baseline["model"]["digest"]

    if digest != expected_digest:
        raise RuntimeError(
            "Repair model digest does not match extraction baseline. "
            f"Expected {expected_digest}, got {digest}."
        )

    return {
        "model": model_name,
        "model_digest": digest,
        "ollama_version": version,
        "options": spec["repair_model"]["options"],
    }


def format_triple(triple):
    subject, predicate, obj = triple
    return f"{predicate}({subject}, {obj})"


def render_current_graph(triples):
    normalized = normalize_triples(triples)

    if not normalized:
        return "(empty graph)"

    return "\n".join(
        format_triple(triple)
        for triple in normalized
    )


def render_allowed_relations(relations):
    return "\n".join(
        f"- {relation}"
        for relation in sorted(relations)
    )


def feedback_for_prompt(feedback):
    rows = sorted(
        feedback,
        key=lambda row: (
            row["validator"],
            row["violation_id"],
        ),
    )
    return json.dumps(
        rows,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def render_repair_prompt(
    template,
    source_text,
    relations,
    triples,
    feedback,
):
    return template.format(
        source_text=source_text,
        allowed_relations=render_allowed_relations(
            relations
        ),
        current_graph=render_current_graph(triples),
        feedback=feedback_for_prompt(feedback),
    )


def parse_repair_response(text, relations):
    allowed = set(relations)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return {
            "ok": False,
            "failure": "empty_output",
            "triples": [],
            "details": None,
        }

    parsed = []

    for line_number, line in enumerate(lines, start=1):
        stripped_line = extract_text2kg.strip_prefix(line)
        call = extract_text2kg.extract_call(stripped_line)

        if call is None:
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {
                    "line_number": line_number,
                    "line": line,
                },
            }

        open_index = stripped_line.find("(")
        depth = 0
        close_index = None

        for index in range(open_index, len(stripped_line)):
            char = stripped_line[index]

            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

                if depth == 0:
                    close_index = index
                    break

        if (
            close_index is None
            or stripped_line[close_index + 1:].strip()
        ):
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {
                    "line_number": line_number,
                    "line": line,
                },
            }

        relation, arguments = call
        pair = extract_text2kg.split_arguments(arguments)

        if pair is None:
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {
                    "line_number": line_number,
                    "line": line,
                },
            }

        subject, obj = pair
        relation = relation.replace("\\_", "_")
        relation = "_".join(relation.strip().split())
        subject = subject.strip()
        obj = obj.strip()

        if not relation or not subject or not obj:
            return {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": {
                    "line_number": line_number,
                    "line": line,
                },
            }

        if relation not in allowed:
            return {
                "ok": False,
                "failure": "relation_outside_allowed_set",
                "triples": [],
                "details": {
                    "line_number": line_number,
                    "relation": relation,
                    "line": line,
                },
            }

        parsed.append((subject, relation, obj))

    normalized = normalize_triples(parsed)

    return {
        "ok": True,
        "failure": None,
        "triples": [
            list(triple) for triple in normalized
        ],
        "details": None,
    }


def generate_repair(prompt, spec):
    response = extract_text2kg.api(
        "/api/generate",
        {
            "model": spec["repair_model"]["name"],
            "prompt": prompt,
            "stream": False,
            "options": spec["repair_model"]["options"],
        },
    )

    return {
        "raw_response": response.get("response", ""),
        "model": response.get("model"),
        "done_reason": response.get("done_reason"),
        "prompt_eval_count": response.get(
            "prompt_eval_count"
        ),
        "eval_count": response.get("eval_count"),
        "total_duration_ns": response.get(
            "total_duration"
        ),
    }


def grounding_identity(triple):
    encoded = json.dumps(
        list(triple),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return f"grounding:{digest}"


def grounding_feedback(triple):
    return {
        "validator": "grounding_v3",
        "violation_id": grounding_identity(triple),
        "error_type": "grounding_error",
        "focus": format_triple(triple),
        "message": (
            "The source sentence does not support this assertion."
        ),
    }


def entity_display_map(context, triples):
    labels = set()

    for case in (
        context["clean"],
        context["injected"],
    ):
        for statement in case.content:
            labels.add(statement.subject)

            if statement.object_kind == "entity":
                labels.add(statement.object)

        for item in case.background_types:
            labels.add(item.entity)

    for subject, _, obj in triples:
        labels.add(subject)
        labels.add(obj)

    output = {}

    from src.controlled_cases import entity_uri, relation_uri, KCL

    for label in labels:
        output[entity_uri(
            context["selected"]["id"],
            label,
        ).n3()] = label

    domain = context["selected"]["domain"]

    for relation in allowed_relations(context):
        if relation in set(
            item.predicate
            for case in (
                context["clean"],
                context["injected"],
            )
            for item in case.content
            if item.object_kind == "date"
        ):
            output[KCL[relation].n3()] = relation
        else:
            try:
                output[
                    relation_uri(domain, relation).n3()
                ] = relation
            except ValueError:
                pass

    return output


def readable_term(value, display_map):
    if value is None:
        return None
    return display_map.get(value, value)


def shacl_feedback_items(
    context,
    triples,
    shacl_validation,
    *,
    primary=False,
):
    display_map = entity_display_map(
        context,
        triples,
    )
    rows = []

    for violation in shacl_validation["violations"]:
        path = readable_term(
            violation.get("result_path"),
            display_map,
        )
        focus = readable_term(
            violation.get("focus_node"),
            display_map,
        )

        rows.append(
            {
                "validator": "raw_shacl",
                "violation_id": violation["violation_id"],
                "error_type": (
                    context["injected"].primary_modification.error_type
                    if (
                        primary
                        and context["injected"].primary_modification
                        is not None
                    )
                    else None
                ),
                "focus": focus,
                "path": path,
                "message": (
                    violation.get("message")
                    or "A SHACL constraint is violated."
                ),
            }
        )

    return rows


def load_frozen_grounding():
    results = index_unique(
        read_jsonl(FROZEN_GROUNDING_RESULTS),
        FROZEN_GROUNDING_RESULTS,
    )
    analysis = read_json(FROZEN_TARGET_ANALYSIS)
    analysis_rows = index_unique(
        analysis["cases"],
        FROZEN_TARGET_ANALYSIS,
    )

    if set(results) != set(analysis_rows):
        raise RuntimeError(
            "Frozen grounding results and target analysis "
            "contain different case ids"
        )

    return results, analysis_rows


def add_judgment(cache, judgment, source):
    triple = judgment.get("triple")
    verdict = judgment.get("verdict")

    if (
        not isinstance(triple, list)
        or len(triple) != 3
        or verdict not in {"SUPPORTED", "UNSUPPORTED"}
    ):
        raise RuntimeError(
            f"Invalid frozen grounding judgment: {judgment!r}"
        )

    key = tuple(triple)
    value = {
        "triple": list(key),
        "verdict": verdict,
        "reason": judgment.get("reason", ""),
        "source": source,
    }

    previous = cache.get(key)

    if (
        previous is not None
        and previous["verdict"] != verdict
    ):
        raise RuntimeError(
            "Frozen grounding results disagree for the same "
            f"assertion: {list(key)}"
        )

    if previous is None:
        cache[key] = value


def grounding_cache_for_case(frozen_row):
    cache = {}

    for state_name in ("clean", "injected"):
        state = frozen_row.get(state_name, {})
        judgments = state.get("judgments")

        if not isinstance(judgments, list):
            raise RuntimeError(
                f"Frozen {state_name} result has no judgments"
            )

        for judgment in judgments:
            add_judgment(
                cache,
                judgment,
                f"frozen_{state_name}",
            )

    clean_unsupported = {
        tuple(row["triple"])
        for row in frozen_row["clean"]["judgments"]
        if row["verdict"] == "UNSUPPORTED"
    }

    return cache, clean_unsupported


def judge_current_grounding(
    source_text,
    triples,
    cache,
):
    rows = []

    for triple in normalize_triples(triples):
        cached = cache.get(triple)

        if cached is None:
            judgment = grounding_judge.judge_triple(
                source_text,
                list(triple),
            )
            cached = {
                "triple": list(triple),
                "verdict": judgment["verdict"],
                "reason": judgment["reason"],
                "source": "repair_round",
                "done_reason": judgment.get(
                    "done_reason"
                ),
                "prompt_eval_count": judgment.get(
                    "prompt_eval_count"
                ),
                "eval_count": judgment.get(
                    "eval_count"
                ),
                "total_duration_ns": judgment.get(
                    "total_duration_ns"
                ),
            }
            cache[triple] = cached

        rows.append(cached)

    return rows


def initial_grounding_feedback(
    context,
    cache,
    target_analysis,
):
    target = target_analysis["target"]

    expected_target_triples = {
        tuple(triple)
        for triple in target["target_triples"]
    }

    delta = primary_delta(
        [
            (
                row.subject,
                row.predicate,
                row.object,
            )
            for row in context["clean"].content
        ],
        [
            (
                row.subject,
                row.predicate,
                row.object,
            )
            for row in context["injected"].content
        ],
    )

    if expected_target_triples != set(delta["added"]):
        if context["selected"]["error_type"] != "cardinality":
            raise RuntimeError(
                f"{context['selected']['id']}: target analysis "
                "does not match controlled content delta"
            )

    feedback = []

    for triple in sorted(expected_target_triples):
        judgment = cache.get(triple)

        if judgment is None:
            raise RuntimeError(
                "Target assertion is missing from frozen grounding "
                f"cache: {list(triple)}"
            )

        if judgment["verdict"] == "UNSUPPORTED":
            feedback.append(
                grounding_feedback(triple)
            )

    observed = bool(feedback)

    if observed != target["observed_grounding_error"]:
        raise RuntimeError(
            f"{context['selected']['id']}: initial grounding "
            "feedback differs from frozen target analysis"
        )

    return feedback


def later_grounding_feedback(
    judgments,
    clean_unsupported,
):
    feedback = []

    for judgment in judgments:
        triple = tuple(judgment["triple"])

        if (
            judgment["verdict"] == "UNSUPPORTED"
            and triple not in clean_unsupported
        ):
            feedback.append(
                grounding_feedback(triple)
            )

    return feedback


def primary_constraint_conforms(context, triples):
    condition = context["selected"]["error_type"]

    if condition not in {"cardinality", "temporal"}:
        return None

    case = repaired_case(context, triples)
    graph = symbolic_graph(context, case)

    return shacl_results(
        graph,
        context["case_shapes"],
    )["conforms"]


def target_resolved(context, triples):
    condition = context["selected"]["error_type"]

    if condition in {"cardinality", "temporal"}:
        return bool(
            primary_constraint_conforms(
                context,
                triples,
            )
        )

    clean = [
        (
            row.subject,
            row.predicate,
            row.object,
        )
        for row in context["clean"].content
    ]
    injected = [
        (
            row.subject,
            row.predicate,
            row.object,
        )
        for row in context["injected"].content
    ]
    added = primary_delta(clean, injected)["added"]
    current = triple_set(triples)

    return all(
        triple not in current
        for triple in added
    )


def evaluate_state(
    context,
    triples,
    cache,
    clean_unsupported,
    *,
    initial,
    target_analysis,
):
    validation = revalidate_symbolic(
        context,
        [
            list(triple)
            for triple in normalize_triples(triples)
        ],
    )

    feedback = shacl_feedback_items(
        context,
        triples,
        validation["shacl"],
        primary=initial,
    )

    owl = owl_feedback(
        context,
        triples,
        validation["owl_consistent"],
    )

    if owl is not None:
        feedback.append(owl)

    if initial:
        normalized_current = normalize_triples(triples)
        missing_frozen = [
            triple
            for triple in normalized_current
            if triple not in cache
        ]

        if missing_frozen:
            raise RuntimeError(
                "Round 0 contains assertions without frozen "
                "grounding judgments: "
                f"{[list(triple) for triple in missing_frozen]}"
            )

        grounding_judgments = [
            cache[triple]
            for triple in normalized_current
        ]
        feedback.extend(
            initial_grounding_feedback(
                context,
                cache,
                target_analysis,
            )
        )
    else:
        grounding_judgments = judge_current_grounding(
            context["clean"].source_text,
            triples,
            cache,
        )
        feedback.extend(
            later_grounding_feedback(
                grounding_judgments,
                clean_unsupported,
            )
        )

    feedback.sort(
        key=lambda row: (
            row["validator"],
            row["violation_id"],
        )
    )

    clean_triples = [
        (
            row.subject,
            row.predicate,
            row.object,
        )
        for row in context["clean"].content
    ]
    injected_triples = [
        (
            row.subject,
            row.predicate,
            row.object,
        )
        for row in context["injected"].content
    ]

    return {
        "symbolic": {
            "shacl": validation["shacl"],
            "owl_consistent": validation[
                "owl_consistent"
            ],
        },
        "grounding": {
            "judgments": grounding_judgments,
            "clean_baseline_unsupported_excluded": [
                list(triple)
                for triple in sorted(clean_unsupported)
            ],
        },
        "actionable_feedback": feedback,
        "target_resolved": target_resolved(
            context,
            triples,
        ),
        "reference": reference_metrics(
            clean_triples,
            injected_triples,
            triples,
        ),
    }


def run_case(
    context,
    frozen_grounding_row,
    target_analysis,
    spec,
    prompt_template,
):
    injected_triples = [
        (
            row.subject,
            row.predicate,
            row.object,
        )
        for row in context["injected"].content
    ]
    clean_triples = [
        (
            row.subject,
            row.predicate,
            row.object,
        )
        for row in context["clean"].content
    ]

    cache, clean_unsupported = (
        grounding_cache_for_case(
            frozen_grounding_row
        )
    )

    initial = evaluate_state(
        context,
        injected_triples,
        cache,
        clean_unsupported,
        initial=True,
        target_analysis=target_analysis,
    )

    initial_feedback = initial[
        "actionable_feedback"
    ]
    initial_ids = {
        row["violation_id"]
        for row in initial_feedback
    }

    rounds = [
        {
            "round": 0,
            "triples": [
                list(triple)
                for triple in normalize_triples(
                    injected_triples
                )
            ],
            "validation": initial,
            "new_violation_ids": [],
        }
    ]

    if not initial_feedback:
        return {
            "id": context["selected"]["id"],
            "domain": context["selected"]["domain"],
            "condition": context["selected"][
                "error_type"
            ],
            "received_initial_feedback": False,
            "initial_feedback_sources": [],
            "rounds": rounds,
            "final": {
                "stop_reason": "no_feedback",
                "repair_rounds": 0,
                "target_resolved": initial[
                    "target_resolved"
                ],
                "validated_state": True,
                "reference_recovery": initial[
                    "reference"
                ]["reference_recovery"],
                "rounds_to_resolution": None,
                "output_failure": None,
            },
        }

    relations = allowed_relations(context)
    current_triples = normalize_triples(
        injected_triples
    )
    current_feedback = initial_feedback
    repair_state_history = []
    first_resolution_round = None
    stop_reason = None
    output_failure = None

    max_rounds = spec["rounds"][
        "max_repair_rounds"
    ]

    for round_number in range(1, max_rounds + 1):
        rendered_prompt = render_repair_prompt(
            prompt_template,
            context["clean"].source_text,
            relations,
            current_triples,
            current_feedback,
        )
        prompt_sha = sha256_bytes(
            rendered_prompt.encode("utf-8")
        )

        generation = generate_repair(
            rendered_prompt,
            spec,
        )

        if generation["done_reason"] == "length":
            parsed = {
                "ok": False,
                "failure": "generation_truncated",
                "triples": [],
                "details": None,
            }
        else:
            parsed = parse_repair_response(
                generation["raw_response"],
                relations,
            )

        repair_record = {
            "rendered_prompt_sha256": prompt_sha,
            "rendered_prompt": rendered_prompt,
            **generation,
            "parse": parsed,
        }

        if not parsed["ok"]:
            output_failure = parsed["failure"]
            rounds.append(
                {
                    "round": round_number,
                    "repair": repair_record,
                    "triples": None,
                    "validation": None,
                    "new_violation_ids": [],
                }
            )
            stop_reason = "output_failure"
            break

        current_triples = normalize_triples(
            parsed["triples"]
        )

        state_result = evaluate_state(
            context,
            current_triples,
            cache,
            clean_unsupported,
            initial=False,
            target_analysis=target_analysis,
        )

        current_feedback = state_result[
            "actionable_feedback"
        ]
        current_ids = {
            row["violation_id"]
            for row in current_feedback
        }

        new_ids = sorted(
            current_ids - initial_ids
        )

        current_state = trajectory_state(
            current_triples,
            current_feedback,
        )

        round_row = {
            "round": round_number,
            "repair": repair_record,
            "triples": [
                list(triple)
                for triple in current_triples
            ],
            "validation": state_result,
            "new_violation_ids": new_ids,
        }
        rounds.append(round_row)

        if (
            state_result["target_resolved"]
            and first_resolution_round is None
        ):
            first_resolution_round = round_number

        stop_reason = stop_reason_after_repair(
            round_number,
            current_state,
            repair_state_history,
            current_feedback,
            max_rounds,
        )

        if stop_reason is not None:
            break

        repair_state_history.append(
            current_state
        )

    final_round = rounds[-1]
    final_validation = final_round.get(
        "validation"
    )

    if final_validation is None:
        target = False
        validated = False
        reference = False
    else:
        target = final_validation[
            "target_resolved"
        ]
        validated = not final_validation[
            "actionable_feedback"
        ]
        reference = final_validation[
            "reference"
        ]["reference_recovery"]

    return {
        "id": context["selected"]["id"],
        "domain": context["selected"]["domain"],
        "condition": context["selected"][
            "error_type"
        ],
        "received_initial_feedback": True,
        "initial_feedback_sources": sorted(
            {
                row["validator"]
                for row in initial_feedback
            }
        ),
        "rounds": rounds,
        "final": {
            "stop_reason": stop_reason,
            "repair_rounds": len(rounds) - 1,
            "target_resolved": target,
            "validated_state": validated,
            "reference_recovery": reference,
            "rounds_to_resolution": (
                first_resolution_round
            ),
            "output_failure": output_failure,
        },
    }


def build_metadata(
    args,
    spec,
    baseline,
    repair_model,
    grounding_model,
):
    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_head": git_head(),
        "repair_spec": str(
            SPEC_PATH.relative_to(ROOT)
        ),
        "repair_spec_sha256": sha256_file(
            SPEC_PATH
        ),
        "repair_prompt": str(
            PROMPT_PATH.relative_to(ROOT)
        ),
        "repair_prompt_sha256": sha256_file(
            PROMPT_PATH
        ),
        "repair_model": repair_model,
        "grounding_model": grounding_model,
        "baseline_manifest": str(
            BASELINE_PATH.relative_to(ROOT)
        ),
        "baseline_manifest_sha256": sha256_file(
            BASELINE_PATH
        ),
        "frozen_grounding_results": str(
            FROZEN_GROUNDING_RESULTS.relative_to(
                ROOT
            )
        ),
        "frozen_grounding_results_sha256": (
            sha256_file(
                FROZEN_GROUNDING_RESULTS
            )
        ),
        "frozen_target_analysis": str(
            FROZEN_TARGET_ANALYSIS.relative_to(
                ROOT
            )
        ),
        "frozen_target_analysis_sha256": (
            sha256_file(
                FROZEN_TARGET_ANALYSIS
            )
        ),
        "max_repair_rounds": spec["rounds"][
            "max_repair_rounds"
        ],
        "start": args.start,
        "limit": args.limit,
        "case_id": args.case_id,
        "output": str(args.output.resolve()),
        "invalid_model_output_retry": False,
        "network_api_retries": 3,
        "notes": [
            (
                "Frozen clean and injected grounding judgments "
                "seed the per-case grounding cache."
            ),
            (
                "Only new assertions encountered during repair "
                "are sent to the frozen grounding assessor."
            ),
            (
                "Malformed or truncated model output is recorded "
                "as an output failure and is not regenerated."
            ),
        ],
    }


def preflight_files():
    required = [
        SPEC_PATH,
        PROMPT_PATH,
        BASELINE_PATH,
        GROUNDING_SPEC_PATH,
        FROZEN_GROUNDING_RESULTS,
        FROZEN_TARGET_ANALYSIS,
    ]

    missing = [
        path for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required repair input files:\n"
            + "\n".join(str(path) for path in missing)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--limit",
        type=int,
    )
    parser.add_argument(
        "--case-id",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify frozen files and local model identities. "
            "No generation or validation case is run."
        ),
    )
    args = parser.parse_args()

    if args.start < 1:
        raise SystemExit("--start must be at least 1")

    preflight_files()
    spec, baseline = load_spec()
    verify_grounding_spec()

    repair_model = local_repair_model_metadata(
        spec,
        baseline,
    )
    grounding_model = (
        grounding_judge.model_metadata()
    )

    if args.preflight_only:
        print("repair preflight: OK")
        print(
            "repair prompt sha256: "
            f"{sha256_file(PROMPT_PATH)}"
        )
        print(
            "repair model digest: "
            f"{repair_model['model_digest']}"
        )
        print(
            "grounding model digest: "
            f"{grounding_model['model_digest']}"
        )
        print(
            "No repair generation or validator case was run."
        )
        return

    if (
        args.output.exists()
        and not args.overwrite
    ):
        raise SystemExit(
            f"Output already exists: {args.output}. "
            "Use --overwrite to replace it."
        )

    frozen_grounding, target_analysis = (
        load_frozen_grounding()
    )

    selection = read_json(
        ROOT / "experiments" / "controlled_selection.json"
    )["cases"]

    if args.case_id is not None:
        selected = [
            row for row in selection
            if row["id"] == args.case_id
        ]

        if len(selected) != 1:
            raise SystemExit(
                f"Unknown controlled case: {args.case_id}"
            )
    else:
        selected = selection[
            args.start - 1:
        ]

        if args.limit is not None:
            selected = selected[:args.limit]

    prompt_template = PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_path = args.output.with_suffix(
        args.output.suffix + ".meta.json"
    )

    metadata = build_metadata(
        args,
        spec,
        baseline,
        repair_model,
        grounding_model,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as output:
        for index, selected_row in enumerate(
            selected,
            start=1,
        ):
            case_id = selected_row["id"]
            print(
                f"[{index:02d}/{len(selected):02d}] "
                f"{case_id} "
                f"({selected_row['error_type']})"
            )

            context = controlled_context(case_id)

            if case_id not in frozen_grounding:
                raise RuntimeError(
                    f"{case_id}: missing frozen grounding result"
                )

            if case_id not in target_analysis:
                raise RuntimeError(
                    f"{case_id}: missing frozen target analysis"
                )

            trajectory = run_case(
                context,
                frozen_grounding[case_id],
                target_analysis[case_id],
                spec,
                prompt_template,
            )

            output.write(
                json.dumps(
                    trajectory,
                    ensure_ascii=False,
                )
                + "\n"
            )
            output.flush()

            final = trajectory["final"]
            print(
                "  stop="
                f"{final['stop_reason']} "
                "target_resolved="
                f"{final['target_resolved']} "
                "reference_recovery="
                f"{final['reference_recovery']} "
                "rounds="
                f"{final['repair_rounds']}"
            )

    metadata["cases"] = len(selected)
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
