"""Script that generates a dump of the MLflow tracking database schema"""

import os
import sys
import tempfile

import sqlalchemy
from sqlalchemy.schema import CreateTable, MetaData

from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


def dump_db_schema(db_url, dst_file):
    engine = sqlalchemy.create_engine(db_url)
    created_tables_metadata = MetaData()
    created_tables_metadata.reflect(bind=engine)
    # Write out table schema as described in
    # https://docs.sqlalchemy.org/en/13/faq/
    # metadata_schema.html#how-can-i-get-the-create-table-drop-table-output-as-a-string
    lines = []
    for ti in created_tables_metadata.sorted_tables:
        for line in str(CreateTable(ti)).splitlines():
            lines.append(line.rstrip() + "\n")
    schema = "".join(lines)
    with open(dst_file, "w") as handle:
        handle.write(schema)


def dump_sqlalchemy_store_schema(dst_file):
    with tempfile.TemporaryDirectory() as db_tmpdir:
        path = os.path.join(db_tmpdir, "db_file")
        db_url = f"sqlite:///{path}"
        SqlAlchemyStore(db_url, db_tmpdir)
        dump_db_schema(db_url, dst_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    dump_sqlalchemy_store_schema(sys.argv[1])
