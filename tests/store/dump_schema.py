"""Script that generates a dump of the SQL schema"""
import os
import shutil
import sys

import sqlalchemy
from sqlalchemy.schema import MetaData
import tempfile

from mlflow.store.sqlalchemy_store import SqlAlchemyStore


def dump_db_schema(metadata, dst_file):
    print("Writing database schema to %s" % dst_file)

    def _dump(sql, *multiparams, **params):  # pylint: disable=unused-argument
        schema = sql.compile(dialect=engine.dialect)
        with open(dst_file, "a") as handle:
            handle.write(str(schema))
    engine = sqlalchemy.create_engine('sqlite://', strategy='mock', executor=_dump)
    metadata.create_all(engine, checkfirst=False)


def dump_sqlalchemy_store_schema(dst_file):
    db_tmpdir = tempfile.mkdtemp()
    path = os.path.join(db_tmpdir, "db_file")
    try:
        db_url = "sqlite:///%s" % path
        SqlAlchemyStore(db_url, db_tmpdir)
        engine = sqlalchemy.create_engine(db_url)
        created_tables_metadata = MetaData(bind=engine)
        created_tables_metadata.reflect()
        dump_db_schema(created_tables_metadata, dst_file)
    finally:
        shutil.rmtree(db_tmpdir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tests/store/dump_schema.py [destination_file]. "
              "Dumps up-to-date database schema to the specified file.")
        sys.exit(1)
    dump_sqlalchemy_store_schema(sys.argv[1])
