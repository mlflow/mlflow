import os
import pandas as pd
import sqlalchemy as sa

from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR

assert _TRACKING_URI_ENV_VAR in os.environ

uri = os.environ[_TRACKING_URI_ENV_VAR]
db_type = uri.split(":")[0]
engine = sa.create_engine(uri)

with engine.begin() as conn:
    for table_name in engine.table_names():
        if db_type in ["postgresql", "mysql"]:
            sql = "SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{}'".format(
                table_name
            )
            df = pd.read_sql(sql, conn)
            df = df.rename(str.lower, axis="columns").set_index("column_name", drop=True)
        elif db_type == "sqlite":
            sql = "PRAGMA table_info('{}')".format(table_name)
            df = pd.read_sql(sql, conn)
            df = df.set_index("name", drop=True)
        else:
            raise ValueError(f"Invalid tracking URI: {uri}")

        for column_name, column_info in df.iterrows():
            new_name = " / ".join([db_type, table_name, column_name])
            print(column_info.fillna("").rename(new_name).to_markdown(tablefmt="grid") + "\n")
