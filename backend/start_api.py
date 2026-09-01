"""Boot the dashboard API with a minimal config.

`main()` insists on a --result artifact, which only exists after a full study
run. The EHR validation, catalog and train endpoints do not need it, so this
constructs the handler directly the way the tests do.
"""
import sys, threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from qhealth_qml.dashboard import _handler

runtime = Path("runs"); runtime.mkdir(exist_ok=True)
config = {
    "runtime_dir": runtime,
    "result_path": runtime / "neutral-result.json",
    "default_model_path": runtime / "neutral-model.pkl",
    "train_lock": threading.Lock(),
    "cors_origin": "*",
}
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
server = ThreadingHTTPServer(("127.0.0.1", port), _handler(config, runtime))
print(f"QML API listening on http://127.0.0.1:{port}", flush=True)
server.serve_forever()
