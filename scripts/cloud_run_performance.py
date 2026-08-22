"""Measure authenticated inference latency for the existing Cloud Run service.

This harness sends only curated ``POST /predict`` requests.  It does not change
Cloud Run configuration and is intentionally a small request/latency probe,
not a high-volume load generator.  Obtain an audience-scoped identity token
from ``gcloud`` or set ``CLOUDSEG_ID_TOKEN`` in the environment; tokens are
never included in the output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class RequestResult:
    """One client-observed prediction request."""

    client_id: int
    request_index: int
    started_at_utc: str
    status_code: int | None
    success: bool
    end_to_end_ms: float
    model_inference_ms: float | None
    error: str | None = None


def percentile(values: list[float], percentile_value: float) -> float | None:
    """Return an interpolated percentile, or ``None`` for an empty sample."""
    if not values:
        return None
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def parse_clients(value: str) -> list[int]:
    """Parse a comma-separated list of positive, distinct concurrency levels."""
    try:
        clients = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("clients must be a comma-separated list of integers") from exc
    if not clients or any(client < 1 for client in clients):
        raise ValueError("clients must contain at least one positive integer")
    if len(set(clients)) != len(clients):
        raise ValueError("clients must not contain duplicate levels")
    return clients


def _normalized_service_url(value: str) -> str:
    service_url = value.rstrip("/")
    parsed = urlparse(service_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("url must be the Cloud Run service root without a path or query")
    return service_url


def identity_token(service_url: str) -> tuple[str, str]:
    """Return a token without exposing it through arguments, files, or output."""
    from_environment = os.environ.get("CLOUDSEG_ID_TOKEN", "").strip()
    if from_environment:
        return from_environment, "CLOUDSEG_ID_TOKEN"

    gcloud = shutil.which(os.environ.get("CLOUDSEG_GCLOUD", "gcloud"))
    if gcloud is None:
        raise RuntimeError(
            "gcloud was not found; set CLOUDSEG_ID_TOKEN or install/authenticate gcloud"
        )
    audience_command = [gcloud, "auth", "print-identity-token", f"--audiences={service_url}"]
    completed = subprocess.run(audience_command, check=False, capture_output=True, text=True)
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip(), "gcloud auth print-identity-token (audience-scoped)"

    # Installed gcloud accepts --audiences for service-account credentials but
    # rejects it for some user-account configurations. Cloud Run accepts the
    # standard user identity token, so retry without the optional audience.
    user_command = [gcloud, "auth", "print-identity-token"]
    fallback = subprocess.run(user_command, check=False, capture_output=True, text=True)
    if fallback.returncode == 0 and fallback.stdout.strip():
        return fallback.stdout.strip(), "gcloud auth print-identity-token (user credentials)"
    detail = fallback.stderr.strip() or completed.stderr.strip() or "no identity token was returned"
    raise RuntimeError(f"could not obtain an identity token with gcloud: {detail}")


def _client(service_url: str, token: str, timeout_seconds: float) -> httpx.Client:
    return httpx.Client(
        base_url=service_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "cloudseg-phase6-performance/1",
        },
        timeout=timeout_seconds,
    )


def _request(
    client: httpx.Client,
    sample_id: str,
    client_id: int,
    request_index: int,
) -> RequestResult:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    try:
        response = client.post("/predict", json={"sample_id": sample_id})
        end_to_end_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            return RequestResult(
                client_id,
                request_index,
                started_at,
                response.status_code,
                False,
                end_to_end_ms,
                None,
                f"HTTP {response.status_code}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            return RequestResult(
                client_id,
                request_index,
                started_at,
                response.status_code,
                False,
                end_to_end_ms,
                None,
                f"invalid JSON response: {exc}",
            )
        model_latency = body.get("latency_ms") if isinstance(body, dict) else None
        if not isinstance(model_latency, (int, float)):
            return RequestResult(
                client_id,
                request_index,
                started_at,
                response.status_code,
                False,
                end_to_end_ms,
                None,
                "response did not contain numeric latency_ms",
            )
        return RequestResult(
            client_id,
            request_index,
            started_at,
            response.status_code,
            True,
            end_to_end_ms,
            float(model_latency),
        )
    except Exception as exc:  # httpx errors are intentionally recorded per request
        return RequestResult(
            client_id,
            request_index,
            started_at,
            None,
            False,
            (time.perf_counter() - started) * 1000.0,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def run_sequential(
    service_url: str,
    token: str,
    sample_id: str,
    count: int,
    timeout_seconds: float,
) -> tuple[list[RequestResult], float]:
    """Run ``count`` predictions over one reused client connection."""
    started = time.perf_counter()
    with _client(service_url, token, timeout_seconds) as client:
        results = [_request(client, sample_id, 0, index) for index in range(count)]
    return results, (time.perf_counter() - started) * 1000.0


def _run_client(
    service_url: str,
    token: str,
    sample_id: str,
    client_id: int,
    count: int,
    timeout_seconds: float,
) -> list[RequestResult]:
    with _client(service_url, token, timeout_seconds) as client:
        return [_request(client, sample_id, client_id, index) for index in range(count)]


def run_concurrency_level(
    service_url: str,
    token: str,
    sample_id: str,
    clients: int,
    requests_per_client: int,
    timeout_seconds: float,
) -> tuple[list[RequestResult], float]:
    """Run independent client loops concurrently against the fixed revision."""
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as executor:
        futures = [
            executor.submit(
                _run_client,
                service_url,
                token,
                sample_id,
                client_id,
                requests_per_client,
                timeout_seconds,
            )
            for client_id in range(clients)
        ]
        results = [result for future in futures for result in future.result()]
    return sorted(results, key=lambda item: (item.client_id, item.request_index)), (
        time.perf_counter() - started
    ) * 1000.0


def summarize(results: list[RequestResult], wall_time_ms: float) -> dict[str, Any]:
    """Summarize successful and failed requests without overstating small samples."""
    successful = [result for result in results if result.success]
    end_to_end = [result.end_to_end_ms for result in successful]
    model_inference = [
        result.model_inference_ms
        for result in successful
        if result.model_inference_ms is not None
    ]
    non_model_overhead = [
        result.end_to_end_ms - result.model_inference_ms
        for result in successful
        if result.model_inference_ms is not None
    ]

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "mean_ms": sum(values) / len(values) if values else None,
            "p50_ms": percentile(values, 50.0),
            "p95_ms": percentile(values, 95.0),
        }

    return {
        "request_count": len(results),
        "success_count": len(successful),
        "failure_count": len(results) - len(successful),
        "wall_time_ms": wall_time_ms,
        "throughput_requests_per_second": (
            len(results) / (wall_time_ms / 1000.0) if wall_time_ms > 0 else None
        ),
        "end_to_end": stats(end_to_end),
        "model_inference": stats(model_inference),
        "end_to_end_minus_model_inference": stats(non_model_overhead),
    }


def _payload(
    service_url: str,
    token_source: str,
    sample_id: str,
    mode: str,
    results: list[RequestResult],
    wall_time_ms: float,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "service_url": service_url,
        "endpoint": f"{service_url}/predict",
        "sample_id": sample_id,
        "token_source": token_source,
        "mode": mode,
        "parameters": parameters,
        "summary": summarize(results, wall_time_ms),
        "requests": [asdict(result) for result in results],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("CLOUDSEG_RUN_URL"),
        help="private Cloud Run service root (or CLOUDSEG_RUN_URL)",
    )
    parser.add_argument("--sample-id", default="Sri-Lanka_713926")
    parser.add_argument(
        "--mode",
        choices=("cold", "sequential", "concurrency"),
        required=True,
        help="cold sends one request; sequential reuses one client; concurrency uses client levels",
    )
    parser.add_argument("--count", type=int, default=20, help="sequential request count")
    parser.add_argument(
        "--clients",
        default="1,2,5",
        help="comma-separated concurrency levels for concurrency mode",
    )
    parser.add_argument(
        "--requests-per-client",
        type=int,
        default=5,
        help="requests issued by each client at each concurrency level",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="per-request timeout in seconds"
    )
    parser.add_argument("--output", type=Path, help="also write the JSON result to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("--url or CLOUDSEG_RUN_URL is required")
    if args.count < 1 or args.requests_per_client < 1 or args.timeout <= 0:
        parser.error("count, requests-per-client, and timeout must be positive")
    try:
        service_url = _normalized_service_url(args.url)
        clients = parse_clients(args.clients)
        token, token_source = identity_token(service_url)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    if args.mode == "cold":
        results, wall_time_ms = run_sequential(service_url, token, args.sample_id, 1, args.timeout)
        parameters = {"count": 1, "idle_before_request": "operator-controlled"}
    elif args.mode == "sequential":
        results, wall_time_ms = run_sequential(
            service_url, token, args.sample_id, args.count, args.timeout
        )
        parameters = {"count": args.count}
    else:
        results = []
        wall_time_ms = 0.0
        level_payloads: list[dict[str, Any]] = []
        for clients_at_level in clients:
            level_results, level_wall_time_ms = run_concurrency_level(
                service_url,
                token,
                args.sample_id,
                clients_at_level,
                args.requests_per_client,
                args.timeout,
            )
            level_payloads.append(
                {
                    "clients": clients_at_level,
                    "requests_per_client": args.requests_per_client,
                    "summary": summarize(level_results, level_wall_time_ms),
                    "requests": [asdict(result) for result in level_results],
                }
            )
            results.extend(level_results)
            wall_time_ms += level_wall_time_ms
        payload = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "service_url": service_url,
            "endpoint": f"{service_url}/predict",
            "sample_id": args.sample_id,
            "token_source": token_source,
            "mode": args.mode,
            "parameters": {
                "clients": clients,
                "requests_per_client": args.requests_per_client,
                "max_instances_changed": False,
                "service_concurrency_changed": False,
            },
            "levels": level_payloads,
        }
        serialized = json.dumps(payload, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if all(result.success for result in results) else 1

    payload = _payload(
        service_url,
        token_source,
        args.sample_id,
        args.mode,
        results,
        wall_time_ms,
        parameters,
    )
    serialized = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
