#!/bin/bash
for BACKOFF in 0 1 2 4 8 16 32 64; do
    if /opt/mssql-tools/bin/sqlcmd -S mssql -U sa -P 1Secure*Password1 -d master -i init-mssql-db.sql; then
        exit 0
    fi
    echo "Could not connect to SQL Server"
    echo "Trying again in ${BACKOFF} seconds"
    sleep $BACKOFF
done
exit 1