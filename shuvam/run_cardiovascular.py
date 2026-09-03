#!/usr/bin/env python
"""Development shim for the cardiovascular frame CLI.

The implementation lives in `qhealth_qml.cardiovascular_cli` and installs as the
`qhealth-cardiovascular` console script; that is what an integrator should use. This
file only exists so the CLI can be run from a source checkout without installing the
package first, which is how it is exercised during development.

    python run_cardiovascular.py status
    qhealth-cardiovascular status        # equivalent, once installed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qhealth_qml.cardiovascular_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
