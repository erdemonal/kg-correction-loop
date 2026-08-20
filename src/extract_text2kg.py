import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "llama3.1:8b-instruct-q4_K_M"

OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 1024,
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


def git_commit(path):
    result = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def model_metadata():
    version = api("/api/version")["version"]
    models = api("/api/tags")["models"]

    model = next(
        (
            item
            for item in models
            if item.get("name") == MODEL or item.get("model") == MODEL
        ),
        None,
    )

    if model is None:
        raise RuntimeError(f"Model not found: {MODEL}")

    return version, model


def strip_prefix(line):
    line = line.strip()
    line = re.sub(r"^[-*•]\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    return line


def extract_call(line):
    line = strip_prefix(line)

    match = re.match(r"([A-Za-z][A-Za-z0-9_\\ ]*)\s*\(", line)
    if not match:
        return None

    relation = match.group(1).strip()
    start = match.end()
    depth = 1

    for index in range(start, len(line)):
        char = line[index]

        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

            if depth == 0:
                return relation, line[start:index]

    return None


def split_arguments(arguments):
    depth = 0

    for index, char in enumerate(arguments):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return arguments[:index], arguments[index + 1:]

    return None


def parse_triples(text):
    triples = []

    for line in text.splitlines():
        call = extract_call(line)

        if call is None:
            continue

        relation, arguments = call
        pair = split_arguments(arguments)

        if pair is None:
            continue

        subject, obj = pair

        relation = relation.replace("\\_", "_")
        relation = re.sub(r"\s+", "_", relation.strip())
        subject = subject.strip()
        obj = obj.strip()

        if not relation or not subject or not obj:
            continue

        triples.append([subject, relation, obj])

    return triples


def deduplicate_triples(triples):
    seen = set()
    unique = []

    for triple in triples:
        key = tuple(triple)

        if key not in seen:
            seen.add(key)
            unique.append(triple)

    return unique


def extract(prompt):
    response = api(
        "/api/generate",
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": OPTIONS,
        },
    )

    raw_response = response.get("response", "")
    parsed_triples_raw = parse_triples(raw_response)
    triples = deduplicate_triples(parsed_triples_raw)

    if response.get("done_reason") == "length":
        status = "truncated"
        failure = "generation reached num_predict"
    else:
        status = "ok"
        failure = None

    return response, parsed_triples_raw, triples, status, failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.start < 1:
        raise SystemExit("--start must be at least 1")

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}. "
            "Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ollama_version, model = model_metadata()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "model_digest": model.get("digest"),
        "ollama_version": ollama_version,
        "options": OPTIONS,
        "generation_format": "Text2KGBench relation(subject, object)",
        "structured_output": False,
        "prompt_modification": False,
        "duplicate_handling": "order-preserving exact duplicate removal",
        "source_prompts": str(args.prompts.resolve()),
        "source_sha256": file_sha256(args.prompts),
        "text2kgbench_commit": git_commit(args.prompts),
        "start": args.start,
        "limit": args.limit,
    }

    metadata_path = args.output.with_suffix(
        args.output.suffix + ".meta.json"
    )

    processed = 0
    truncated = 0
    api_errors = 0

    with args.prompts.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            if line_number < args.start:
                continue

            if args.limit is not None and processed >= args.limit:
                break

            item = json.loads(line)

            try:
                response, parsed_triples_raw, triples, status, failure = extract(
                    item["prompt"]
                )
            except Exception as exc:
                response = {}
                parsed_triples_raw = []
                triples = []
                status = "api_error"
                failure = f"{type(exc).__name__}: {exc}"
                api_errors += 1

            if status == "truncated":
                truncated += 1

            result = {
                "id": item["id"],
                "status": status,
                "error": failure,
                "response": response.get("response"),
                "parsed_triples_raw": parsed_triples_raw,
                "triples": triples,
                "model": response.get("model", MODEL),
                "created_at": response.get("created_at"),
                "done_reason": response.get("done_reason"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "total_duration_ns": response.get("total_duration"),
            }

            output.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output.flush()

            processed += 1

            print(
                f"{processed}: {item['id']} -> "
                f"{len(triples)} triples [{status}]"
            )

    metadata["cases"] = processed
    metadata["truncated"] = truncated
    metadata["api_errors"] = api_errors

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Cases: {processed}")
    print(f"Truncated: {truncated}")
    print(f"API errors: {api_errors}")
    print(f"Output: {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
