# ruff: noqa: T201
"""
Verify a fs2db migration by printing row counts from the target DB.
"""

GREEN = "\033[32m"
RESET = "\033[0m"

_COUNT_QUERIES: dict[str, str] = {
    "experiments": "SELECT COUNT(*) FROM experiments",
    "runs": "SELECT COUNT(*) FROM runs",
    "params": "SELECT COUNT(*) FROM params",
    "tags": "SELECT COUNT(*) FROM tags",
    "metrics": "SELECT COUNT(*) FROM metrics",
    "datasets": "SELECT COUNT(*) FROM datasets",
    "inputs": (
        "SELECT COUNT(*) FROM inputs WHERE source_type = 'DATASET' AND destination_type = 'RUN'"
    ),
    "outputs": (
        "SELECT COUNT(*) FROM inputs"
        " WHERE source_type = 'RUN_OUTPUT' AND destination_type = 'MODEL_OUTPUT'"
    ),
    "traces": "SELECT COUNT(*) FROM trace_info",
    "assessments": "SELECT COUNT(*) FROM assessments",
    "logged_models": "SELECT COUNT(*) FROM logged_models",
    "registered_models": "SELECT COUNT(*) FROM registered_models",
    "model_versions": "SELECT COUNT(*) FROM model_versions",
}


def verify_migration(target_uri: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(target_uri)
    print()
    print("Row counts:")
    with engine.connect() as conn:
        for entity, query in _COUNT_QUERIES.items():
            try:
                count = conn.execute(text(query)).scalar()
            except Exception:
                count = 0
            print(f"  {GREEN}{entity}{RESET}: {count}")
    print()
