import { describe, expect, test } from '@jest/globals';
import { render as rtlRender, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ASSESSMENT_SESSION_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { CellContext } from '@tanstack/react-table';
import { buildAssessmentColumnDefs } from './buildAssessmentColumnDefs';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';

describe('buildAssessmentColumnDefs', () => {
  test('builds column defs for assessment columns with correct ids and labelText', () => {
    const columns = [
      { name: 'relevance', type: 'categorical' as const },
      { name: 'score', type: 'numeric' as const },
    ];
    const defs = buildAssessmentColumnDefs(columns);

    expect(defs).toHaveLength(2);
    expect(defs[0].id).toBe('assessment:relevance');
    expect(defs[1].id).toBe('assessment:score');
    // labelText is set for a11y on the column menu trigger.
    expect(defs[0].labelText).toBe('relevance');
    expect(defs[1].labelText).toBe('score');
  });

  test('renders assessment column header with gavel icon', () => {
    const columns = [{ name: 'relevance', type: 'categorical' as const }];
    const defs = buildAssessmentColumnDefs(columns);

    const headerElement = defs[0].header as () => JSX.Element;
    const { container } = rtlRender(headerElement());

    expect(container.querySelector('[data-testid="trace-assessment-column-icon"]')).toBeInTheDocument();
    expect(container.textContent).toContain('relevance');
  });

  test('shows the source icon when the column has a single source', () => {
    const defs = buildAssessmentColumnDefs([{ name: 'tone', type: 'categorical' as const, sourceType: 'HUMAN' }]);

    const headerElement = defs[0].header as () => JSX.Element;
    const { container } = rtlRender(headerElement());

    expect(container.querySelector('[data-testid="trace-assessment-column-human-icon"]')).toBeInTheDocument();
    expect(container.querySelector('[data-testid="trace-assessment-column-icon"]')).not.toBeInTheDocument();
  });

  test('assigns correct column sizes', () => {
    const columns = [{ name: 'test', type: 'categorical' as const }];
    const defs = buildAssessmentColumnDefs(columns);

    expect(defs[0].size).toBe(160);
    expect(defs[0].minSize).toBe(100);
    expect(defs[0].maxSize).toBe(480);
  });
});

describe('buildAssessmentColumnDefs session-level columns', () => {
  const renderNode = (node: React.ReactNode) =>
    rtlRender(
      <IntlProvider locale="en">
        <DesignSystemProvider>{node}</DesignSystemProvider>
      </IntlProvider>,
    );
  const cellFor = (trace: ModelTraceInfoV3) => ({ row: { original: trace } }) as CellContext<ModelTraceInfoV3, unknown>;
  const attachmentTrace = makeTrace('turn-1', {
    assessments: [
      makeFeedbackAssessment('completeness', 'complete-value', {
        metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session-1' },
      }),
    ],
  });
  const otherTurn = makeTrace('turn-2', { assessments: [] });
  const columns = [{ name: 'completeness', type: 'categorical' as const }];
  const sessionAssessmentNames = new Set(['completeness']);

  test('flat: shows the session value on the trace it is attached to', () => {
    const [def] = buildAssessmentColumnDefs(columns, { sessionAssessmentNames, isGroupedBySession: false });
    const cell = def.cell as (ctx: CellContext<ModelTraceInfoV3, unknown>) => React.ReactNode;
    renderNode(cell(cellFor(attachmentTrace)));
    expect(screen.getByText('complete-value')).toBeInTheDocument();
    expect(cell(cellFor(otherTurn))).toBeNull();
  });

  test('grouped: shows the value once on the session row and leaves the turns empty', () => {
    const [def] = buildAssessmentColumnDefs(columns, { sessionAssessmentNames, isGroupedBySession: true });
    const cell = def.cell as (ctx: CellContext<ModelTraceInfoV3, unknown>) => React.ReactNode;
    expect(cell(cellFor(attachmentTrace))).toBeNull();
    renderNode(def.renderSessionCell?.([otherTurn, attachmentTrace]));
    expect(screen.getByText('complete-value')).toBeInTheDocument();
  });

  test('without session names, a column has no session cell', () => {
    const [def] = buildAssessmentColumnDefs(columns);
    expect(def.renderSessionCell).toBeUndefined();
  });
});
