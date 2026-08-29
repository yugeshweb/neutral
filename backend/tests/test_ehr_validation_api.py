"""Focused checks for the isolated EHR cohort validation endpoint."""

from __future__ import annotations

import json
import threading
from http.client import HTTPResponse
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from http.server import ThreadingHTTPServer

from qhealth_qml.dashboard import _handler


def _post(server: ThreadingHTTPServer, body: dict) -> tuple[int, dict]:
    request = Request(
        f"http://127.0.0.1:{server.server_port}/api/validation/ehr",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response: HTTPResponse = urlopen(request)
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))
    return response.status, json.loads(response.read().decode("utf-8"))


def _server(runtime_dir: Path) -> ThreadingHTTPServer:
    config = {"runtime_dir": runtime_dir, "cors_origin": "*"}
    return ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler(config, runtime_dir),
    )


def test_ehr_validation_accepts_canonical_cohort_and_previews_routing() -> None:
    csv_text = "patient_id,feature_a,label\np1,1,0\np2,2,1\np3,3,0\np4,4,1\n"

    with TemporaryDirectory() as directory:
        runtime_path = Path(directory)
        server = _server(runtime_path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, payload = _post(
                server,
                {
                    "csv_text": csv_text,
                    "dataset_name": "small-neuro-cohort.csv",
                    "source_format": "csv",
                    "target": "label",
                    "id_column": "patient_id",
                },
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        persistent_uploads = (runtime_path / "bundle_uploads").exists()

    assert status == 200
    assert payload["status"] == "passed"
    assert payload["dataset"]["name"] == "small-neuro-cohort.csv"
    assert payload["dataset"]["rows"] == 4
    assert payload["dataset"]["features"] == 1
    assert payload["dataset"]["missing_cells"] == 0
    assert payload["bundle"]["assets"][0]["validation_status"] == "accepted"
    assert payload["bundle"]["assets"][0]["uri"] == "ephemeral://ehr-validation"
    assert payload["routing"]
    assert all("status" in decision and decision["reason"] for decision in payload["routing"])
    assert "No model was trained" in payload["disclaimer"]
    assert not persistent_uploads


def test_ehr_validation_rejects_missing_canonical_csv() -> None:
    with TemporaryDirectory() as directory:
        server = _server(Path(directory))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, payload = _post(server, {"target": "label"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    assert status == 400
    assert "csv_text is required" in payload["error"]
