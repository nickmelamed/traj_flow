"""Console-script wrappers for the two Streamlit apps.

Streamlit apps are plain scripts executed via `streamlit run <file>`, not
importable `main()` callables, so `trajflow-review-app` / `trajflow-dashboard`
just shell out to `streamlit run` against this package's installed copy of
the script -- giving them the same one-command feel as every other pipeline
stage without duplicating Streamlit's own CLI.
"""

import sys
from pathlib import Path


def _run_streamlit(script: Path) -> None:
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(script), *sys.argv[1:]]
    sys.exit(stcli.main())


def run_review_app() -> None:
    from trajflow.hitl import review_app

    _run_streamlit(Path(review_app.__file__))


def run_dashboard() -> None:
    from trajflow.viz import dashboard

    _run_streamlit(Path(dashboard.__file__))
