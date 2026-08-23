import hashlib
import json
import time
from pathlib import Path
from urllib import error, request


OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b-instruct-q4_K_M"
EXPECTED_DIGEST = (
    "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
)
JUDGE_VERSION = "v3"

OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 256,
}

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "experiments" / "grounding_judge_prompt_v3.txt"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "UNSUPPORTED"],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def api(path, payload=None, retries=3):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    for attempt in range(1, retries + 1):
        req = request.Request(
            OLLAMA_URL + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )

        try:
            with request.urlopen(req, timeout=600) as response:
                return json.load(response)
        except (error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(1)


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def model_metadata():
    version = api("/api/version")["version"]
    models = api("/api/tags")["models"]

    model = next(
        (
            item
            for item in models
            if item.get("name") == MODEL
            or item.get("model") == MODEL
        ),
        None,
    )

    if model is None:
        raise RuntimeError(f"Model not found: {MODEL}")

    digest = model.get("digest")

    if digest != EXPECTED_DIGEST:
        raise RuntimeError(
            "Grounding judge digest does not match the locked model. "
            f"Expected {EXPECTED_DIGEST}, got {digest}."
        )

    return {
        "judge_version": JUDGE_VERSION,
        "ollama_version": version,
        "model": MODEL,
        "model_digest": digest,
        "options": OPTIONS,
        "prompt_path": str(PROMPT_PATH.relative_to(ROOT)),
        "prompt_sha256": file_sha256(PROMPT_PATH),
    }


def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8")


def render_prompt(source_text, triple, template=None):
    if template is None:
        template = load_prompt()

    if (
        not isinstance(triple, list)
        or len(triple) != 3
        or not all(isinstance(value, str) for value in triple)
    ):
        raise ValueError(f"Invalid triple: {triple!r}")

    subject, predicate, obj = triple

    return template.format(
        source_text=source_text,
        subject=subject,
        predicate=predicate,
        object=obj,
    )


def parse_judgment(text):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Grounding judge returned invalid JSON: {text!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Grounding judgment must be a JSON object")

    if set(payload) != {"verdict", "reason"}:
        raise RuntimeError(
            "Grounding judgment must contain only verdict and reason"
        )

    verdict = payload["verdict"]
    reason = payload["reason"]

    if verdict not in {"SUPPORTED", "UNSUPPORTED"}:
        raise RuntimeError(f"Invalid grounding verdict: {verdict!r}")

    if not isinstance(reason, str):
        raise RuntimeError("Grounding reason must be a string")

    return {
        "verdict": verdict,
        "reason": reason.strip(),
    }


def judge_triple(source_text, triple, template=None):
    prompt = render_prompt(source_text, triple, template=template)

    response = api(
        "/api/generate",
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": OPTIONS,
        },
    )

    raw = response.get("response", "")
    judgment = parse_judgment(raw)

    return {
        "triple": triple,
        "verdict": judgment["verdict"],
        "reason": judgment["reason"],
        "done_reason": response.get("done_reason"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "total_duration_ns": response.get("total_duration"),
    }


def aggregate_case(triple_judgments):
    return any(
        row["verdict"] == "UNSUPPORTED"
        for row in triple_judgments
    )


def judge_case(source_text, triples, template=None):
    judgments = [
        judge_triple(source_text, triple, template=template)
        for triple in triples
    ]

    return {
        "grounding_error": aggregate_case(judgments),
        "unsupported_count": sum(
            row["verdict"] == "UNSUPPORTED"
            for row in judgments
        ),
        "triple_count": len(judgments),
        "judgments": judgments,
    }
