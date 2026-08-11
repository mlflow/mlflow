import { DatabaseIcon, Empty } from '@databricks/design-system';
import { FormattedMessage, type IntlShape } from 'react-intl';
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

/**
 * V4 search needs a SQL warehouse; prompt the user to pick one rather than firing a doomed query.
 * Product-specific, so it stays MLflow-side and is passed to `TracesTableView` as `customEmptyState`.
 * Centered to match the shared states' layout.
 */
export const TracesV4NoWarehouseState = () => (
  <div
    css={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      minHeight: 360,
      width: '100%',
      '& > div': {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
      },
    }}
  >
    <Empty
      image={<DatabaseIcon />}
      title={
        <FormattedMessage
          defaultMessage="Select a SQL warehouse"
          description="Title for the state shown when no SQL warehouse is selected for V4 traces"
        />
      }
      description={
        <FormattedMessage
          defaultMessage="Choose a SQL warehouse to query traces stored in Unity Catalog."
          description="Description for the no-SQL-warehouse state on the V4 traces tab"
        />
      }
    />
  </div>
);
