import type { TraceTableColumn } from '@databricks/web-shared/traces-table';
import { TraceAssessmentCell } from '../components/TraceAssessmentCell';
import { assessmentColumnId } from './assessmentColumns';

// Seed widths for assessment columns — comfortable for a tag, resizable within these bounds.
const ASSESSMENT_COLUMN_SIZE = { size: 160, minSize: 100, maxSize: 480 } as const;

/** Build one TanStack column def per visible assessment name (canonical sorted order). */
export const buildAssessmentColumnDefs = (names: string[]): TraceTableColumn[] =>
  names.map((name) => ({
    id: assessmentColumnId(name),
    ...ASSESSMENT_COLUMN_SIZE,
    header: () => name,
    cell: (ctx) => <TraceAssessmentCell trace={ctx.row.original} assessmentName={name} />,
  }));
