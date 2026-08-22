from __future__ import annotations

from scripts import cloud_run_performance as performance


def test_percentile_interpolates_and_handles_empty_values() -> None:
    assert performance.percentile([], 50) is None
    assert performance.percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert performance.percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def test_parse_clients_rejects_invalid_levels() -> None:
    assert performance.parse_clients("1, 2,5") == [1, 2, 5]
    for value in ("", "0,1", "1,1", "one"):
        try:
            performance.parse_clients(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {value!r}")


def test_summary_separates_model_and_end_to_end_latency() -> None:
    results = [
        performance.RequestResult(0, 0, "now", 200, True, 120.0, 100.0),
        performance.RequestResult(0, 1, "now", 200, True, 180.0, 140.0),
        performance.RequestResult(0, 2, "now", 500, False, 20.0, None, "HTTP 500"),
    ]
    summary = performance.summarize(results, 300.0)
    assert summary["request_count"] == 3
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["end_to_end"]["mean_ms"] == 150.0
    assert summary["model_inference"]["p50_ms"] == 120.0
    assert summary["end_to_end_minus_model_inference"]["mean_ms"] == 30.0
