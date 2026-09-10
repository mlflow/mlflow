import { FormattedMessage } from '@databricks/i18n';
import { ISSUES_COLUMN_ID } from '@databricks/web-shared/genai-traces-table';
import type { TraceTableColumn } from '@databricks/web-shared/traces-table';
// Not re-exported from the genai-traces-table barrel — import from its module.
import { IssuesCell } from '@databricks/web-shared/genai-traces-table/cellRenderers/IssuesCell';
import { extractTraceIssues } from './assessmentColumns';

// Seed width for the Issues column — comfortable for a coral tag, resizable within these bounds.
const ISSUES_COLUMN_SIZE = { size: 160, minSize: 100, maxSize: 480 } as const;

/**
 * The dedicated "Issues" column (mirrors the prior tab): detected issues render as coral tags that
 * link to the issue detail page, kept separate from assessment columns. Placed ahead of the dynamic
 * assessment columns via `extraColumns`.
 */
export const buildIssuesColumnDef = (): TraceTableColumn => ({
  id: ISSUES_COLUMN_ID,
  ...ISSUES_COLUMN_SIZE,
  header: () => (
    <FormattedMessage defaultMessage="Issues" description="Column label for the traces table issues column" />
  ),
  cell: (ctx) => (
    <IssuesCell issues={extractTraceIssues(ctx.row.original)} otherIssues={undefined} isComparing={false} />
  ),
});
