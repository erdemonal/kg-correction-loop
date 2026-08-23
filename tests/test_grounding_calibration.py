from src.prepare_grounding_calibration import (
    eligible_rows,
    split_rows,
)


def rows():
    output = []

    for domain in ("movie", "music"):
        for index in range(6):
            output.append(
                {
                    "id": f"{domain}-{index}",
                    "domain": domain,
                    "annotated": True,
                    "parse_issue": False,
                    "labels": (
                        ["grounding_error"]
                        if index % 2
                        else []
                    ),
                }
            )

    output.append(
        {
            "id": "excluded",
            "domain": "movie",
            "annotated": True,
            "parse_issue": True,
            "labels": ["grounding_error"],
        }
    )

    return output


def test_parse_issue_rows_are_excluded():
    eligible = eligible_rows(rows())

    assert len(eligible) == 12
    assert all(row["id"] != "excluded" for row in eligible)


def test_split_is_deterministic_disjoint_and_complete():
    first_cal, first_hold = split_rows(rows())
    second_cal, second_hold = split_rows(rows())

    assert first_cal == second_cal
    assert first_hold == second_hold

    cal_ids = {row["id"] for row in first_cal}
    hold_ids = {row["id"] for row in first_hold}

    assert not cal_ids & hold_ids
    assert len(cal_ids | hold_ids) == 12


def test_each_nontrivial_stratum_has_heldout_examples():
    calibration, heldout = split_rows(rows())

    for domain in ("movie", "music"):
        for human in (False, True):
            assert any(
                row["domain"] == domain
                and row["human_grounding_error"] == human
                for row in calibration
            )
            assert any(
                row["domain"] == domain
                and row["human_grounding_error"] == human
                for row in heldout
            )
