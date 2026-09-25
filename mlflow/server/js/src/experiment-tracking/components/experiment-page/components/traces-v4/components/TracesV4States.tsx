import { type IntlShape } from 'react-intl';
import { isSqlWarehouseTimeoutError } from '@databricks/web-shared/genai-traces-table';

/**
 * Turn a failed search error into a user-facing description for the shared error surfaces
 * (`TracesErrorState` / `TracesErrorAlert` accept this via `getErrorDescription`). A SQL-warehouse
 * timeout gets the actionable "use a larger warehouse" hint (matching v3); otherwise the raw server
 * message so a permission error reads distinctly; else a generic fallback. MLflow owns this text
 * because the SQL-warehouse concept is product-specific.
 */
export const makeTracesV4ErrorDescription =
  (intl: IntlShape) =>
  (error: unknown): string => {
    const err = error instanceof Error ? error : undefined;
    if (isSqlWarehouseTimeoutError(err)) {
      return intl.formatMessage({
        defaultMessage:
          'The SQL query timed out. Please retry, and if the problem persists, try selecting a larger SQL warehouse.',
        description: 'V4 traces > SQL warehouse timeout error description with CTA to select a larger warehouse',
      });
    }
    return (
      err?.message ||
      intl.formatMessage({
        defaultMessage: 'Something went wrong while loading traces.',
        description: 'Generic fallback description for the V4 traces load-error surfaces',
      })
    );
  };

// Note: the Databricks build also has a `TracesV4NoWarehouseState` ("select a SQL warehouse" empty
// state). OSS has no SQL warehouse / Unity Catalog, so that state is unreachable and is dropped here;
// the search always fires against the MLflow experiment location.
