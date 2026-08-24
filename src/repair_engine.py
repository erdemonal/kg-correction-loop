import hashlib
import json


def normalize_triples(triples):
    seen = set()
    output = []

    for triple in triples:
        if (
            not isinstance(triple, (list, tuple))
            or len(triple) != 3
            or not all(isinstance(value, str) for value in triple)
        ):
            raise ValueError(f"Invalid triple: {triple!r}")

        value = tuple(triple)

        if value not in seen:
            seen.add(value)
            output.append(value)

    return tuple(output)


def triple_set(triples):
    return frozenset(normalize_triples(triples))


def graph_state(triples):
    canonical = sorted(triple_set(triples))
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def violation_identity_set(feedback):
    ids = []

    for item in feedback:
        violation_id = item.get("violation_id")

        if not isinstance(violation_id, str) or not violation_id:
            raise ValueError(
                f"Feedback item has no violation_id: {item!r}"
            )

        ids.append(violation_id)

    return frozenset(ids)


def trajectory_state(triples, feedback):
    return (
        graph_state(triples),
        tuple(sorted(violation_identity_set(feedback))),
    )


def primary_delta(clean_triples, injected_triples):
    clean = triple_set(clean_triples)
    injected = triple_set(injected_triples)

    return {
        "added": frozenset(injected - clean),
        "removed": frozenset(clean - injected),
    }


def reference_metrics(
    clean_triples,
    injected_triples,
    current_triples,
):
    clean = triple_set(clean_triples)
    injected = triple_set(injected_triples)
    current = triple_set(current_triples)

    missing = frozenset(clean - current)
    extra = frozenset(current - clean)

    primary = primary_delta(clean, injected)
    collateral_removed = frozenset(
        missing - primary["removed"]
    )
    collateral_added = frozenset(
        extra - primary["added"]
    )

    return {
        "reference_recovery": current == clean,
        "clean_reference_removed": [
            list(triple) for triple in sorted(missing)
        ],
        "new_not_in_clean_reference": [
            list(triple) for triple in sorted(extra)
        ],
        "reference_symmetric_difference": len(
            missing | extra
        ),
        "collateral_removed": [
            list(triple)
            for triple in sorted(collateral_removed)
        ],
        "collateral_added": [
            list(triple)
            for triple in sorted(collateral_added)
        ],
        "collateral_symmetric_difference": len(
            collateral_removed | collateral_added
        ),
    }


def stop_reason_after_repair(
    round_number,
    current_state,
    repair_state_history,
    actionable_feedback,
    max_rounds,
):
    if round_number < 1:
        raise ValueError("round_number must be at least 1")

    if not actionable_feedback:
        return "validated"

    if round_number >= 2:
        previous_state = repair_state_history[-1]

        if current_state == previous_state:
            return "stalled"

    if round_number >= 3:
        earlier_nonadjacent = repair_state_history[:-1]

        if current_state in earlier_nonadjacent:
            return "oscillation"

    if round_number >= max_rounds:
        return "max_rounds"

    return None
