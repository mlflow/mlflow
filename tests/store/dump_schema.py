"""Script that generates a dump of the MLflow tracking database schema"""
import os
import shutil
import sys

import sqlalchemy
from sqlalchemy.schema import CreateTable, MetaData
import tempfile

from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


def dump_db_schema(db_url, dst_file):
    engine = sqlalchemy.create_engine(db_url)
    created_tables_metadata = MetaData(bind=engine)
    created_tables_metadata.reflect()
    # Write out table schema as described in
    # https://docs.sqlalchemy.org/en/13/faq/
    # metadata_schema.html#how-can-i-get-the-create-table-drop-table-output-as-a-string
    lines = []
    for ti in created_tables_metadata.sorted_tables:
        for line in str(CreateTable(ti)).splitlines():
            lines.append(line.rstrip() + "\n")
    schema = "".join(lines)
    print("Writing database schema to %s" % dst_file)
    with open(dst_file, "w") as handle:
        handle.write(schema)


def dump_sqlalchemy_store_schema(dst_file):
    db_tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(db_tmpdir, "db_file")
        db_url = "sqlite:///%s" % path
        SqlAlchemyStore(db_url, db_tmpdir)
        dump_db_schema(db_url, dst_file)
    finally:
        shutil.rmtree(db_tmpdir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: python tests/store/dump_schema.py [destination_file]. "
            "Dumps up-to-date database schema to the specified file."
        )
        sys.exit(1)
    dump_sqlalchemy_store_schema(sys.argv[1])
