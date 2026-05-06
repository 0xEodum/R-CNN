from src.eval_checkpoint import parse_score_thresholds


def test_parse_score_thresholds_accepts_comma_separated_values() -> None:
    assert parse_score_thresholds("0.3, 0.4,0.50") == (0.3, 0.4, 0.5)


def test_parse_score_thresholds_rejects_empty_values() -> None:
    try:
        parse_score_thresholds(" , ")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Expected empty score threshold list to fail")
