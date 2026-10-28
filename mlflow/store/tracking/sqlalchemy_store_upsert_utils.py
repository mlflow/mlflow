from __future__ import annotations

from typing import Any

import sqlalchemy
from sqlalchemy.orm import Session


def _bulk_upsert(session: Session, model_class: type, rows: list[dict[str, Any]]) -> None:
    """Bulk upsert rows using dialect-specific INSERT ON CONFLICT.

    Rows are inserted in batches to stay within database limits (e.g., SQLite's
    SQLITE_MAX_VARIABLE_NUMBER on older versions, MySQL's max_allowed_packet).

    Falls back to per-row session.merge() for unsupported dialects (e.g., MSSQL).
    """
    if not rows:
        return

    dialect = session.bind.dialect.name
    table = model_class.__table__
    pk_columns = [col.name for col in table.primary_key.columns]
    # Only update columns present in the inserted rows. Updating omitted columns
    # references missing insert aliases on MySQL (`new.<col>`) and can overwrite
    # columns that are intentionally absent from the incoming rows.
    provided_columns = set.intersection(*(set(row) for row in rows))
    update_columns = [
        c.name
        for c in table.columns
        if c.name not in pk_columns and not c.computed and c.name in provided_columns
    ]

    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        _upsert_batch(session, model_class, table, batch, pk_columns, update_columns, dialect)


def _upsert_batch(
    session: Session,
    model_class: type,
    table: sqlalchemy.Table,
    rows: list[dict[str, Any]],
    pk_columns: list[str],
    update_columns: list[str],
    dialect: str,
) -> None:
    match dialect:
        case "sqlite" | "postgresql":
            if dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert
            else:
                from sqlalchemy.dialects.postgresql import insert

            stmt = insert(table).values(rows)
            if update_columns:
                stmt = stmt.on_conflict_do_update(
                    index_elements=pk_columns,
                    set_={col: stmt.excluded[col] for col in update_columns},
                )
            else:
                stmt = stmt.on_conflict_do_nothing()
            session.execute(stmt)
        case "mysql":
            from sqlalchemy.dialects.mysql import insert

            stmt = insert(table).values(rows)
            if update_columns:
                stmt = stmt.on_duplicate_key_update({
                    col: stmt.inserted[col] for col in update_columns
                })
            else:
                # No-op update on PK to silently skip duplicates
                stmt = stmt.on_duplicate_key_update({pk_columns[0]: stmt.inserted[pk_columns[0]]})
            session.execute(stmt)
        case _:
            # Fallback for MSSQL and other dialects
            for row in rows:
                session.merge(model_class(**row))
