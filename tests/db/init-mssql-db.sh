#!/bin/bash
sleep 60 # wait for SQL Server to startup
/opt/mssql-tools/bin/sqlcmd -S mssql -U sa -P 1Secure*Password1 -d master -i init-mssql-db.sql