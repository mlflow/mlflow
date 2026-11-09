import type { TraceTableColumn } from '@databricks/web-shared/traces-table';
import { AssessmentSourceIcon } from '../components/AssessmentSourceIcon';
import { AssessmentCell, TraceAssessmentCell } from '../components/TraceAssessmentCell';
import { assessmentColumnId, pickSessionCellAssessment } from './assessmentColumns';

// Seed widths for assessment columns — comfortable for a tag, resizable within these bounds.
const ASSESSMENT_COLUMN_SIZE = { size: 160, minSize: 100, maxSize: 480 } as const;

export interface AssessmentColumn {
  name: string;
  type: 'numeric' | 'categorical';
  /** Set when every assessment shown in the column has this source; the header then shows its icon. */
  sourceType?: string;
}

export interface AssessmentColumnDefsOptions {
  /**
   * Names of session-level assessments. Their columns show the session's value once per session row
   * when grouped, and on the trace the assessment is attached to otherwise.
   */
  sessionAssessmentNames?: ReadonlySet<string>;
  isGroupedBySession?: boolean;
}

/** Build one TanStack column def per visible assessment (canonical sorted order). */
export const buildAssessmentColumnDefs = (
  assessmentColumns: AssessmentColumn[],
  { sessionAssessmentNames, isGroupedBySession = false }: AssessmentColumnDefsOptions = {},
): TraceTableColumn[] =>
  assessmentColumns.map(({ name, type, sourceType }) => {
    const header = () => (
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
        <AssessmentSourceIcon sourceType={sourceType} testIdPrefix="trace-assessment-column" />
        {name}
      </span>
    );
    if (sessionAssessmentNames?.has(name)) {
      return {
        id: assessmentColumnId(name),
        ...ASSESSMENT_COLUMN_SIZE,
        header,
        labelText: name,
        cell: (ctx) => {
          if (isGroupedBySession) {
            return null;
          }
          const assessment = pickSessionCellAssessment([ctx.row.original], name);
          return assessment ? <AssessmentCell assessment={assessment} columnType={type} /> : null;
        },
        renderSessionCell: (traces) => {
          const assessment = pickSessionCellAssessment(traces, name);
          return assessment ? <AssessmentCell assessment={assessment} columnType={type} /> : null;
        },
      };
    }
    return {
      id: assessmentColumnId(name),
      ...ASSESSMENT_COLUMN_SIZE,
      header,
      labelText: name,
      cell: (ctx) => <TraceAssessmentCell trace={ctx.row.original} assessmentName={name} columnType={type} />,
    };
  });
