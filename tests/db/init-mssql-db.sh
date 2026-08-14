#!/bin/bash
for BACKOFF in 0 1 2 4 8 16; do
    if [ "$BACKOFF" -ne "0" ]; then
        echo "Could not connect to SQL Server"
        echo "Trying again in ${BACKOFF} seconds"
        sleep $BACKOFF
    fi
    if /opt/mssql-tools/bin/sqlcmd -S mssql -U sa -P 1Secure*Password1 -d master -i $(dirname "$0")/init-mssql-db.sql; then
        exit 0
    fi
done
exit 1
