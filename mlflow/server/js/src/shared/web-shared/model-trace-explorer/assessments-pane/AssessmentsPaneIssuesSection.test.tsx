import { describe, expect, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AssessmentsPaneIssuesSection } from './AssessmentsPaneIssuesSection';
import type { IssueReferenceAssessment } from '../ModelTrace.types';

const reference = (
  overrides: Partial<IssueReferenceAssessment> & Pick<IssueReferenceAssessment, 'assessment_id'>,
): IssueReferenceAssessment => ({
  assessment_name: 'iss-latency',
  trace_id: 'trace-1',
  span_id: 'span-1',
  source: { source_type: 'LLM_JUDGE', source_id: 'mlflow.issue_detection' },
  create_time: '2026-10-05T05:35:58Z',
  last_update_time: '2026-10-05T05:35:58Z',
  issue: { issue_name: 'High latency' },
  rationale: 'The tool took too long.',
  ...overrides,
});

const renderSection = (issues: IssueReferenceAssessment[]) =>
  render(
    <DesignSystemProvider>
      <IntlProvider locale="en">
        <AssessmentsPaneIssuesSection issues={issues} />
      </IntlProvider>
    </DesignSystemProvider>,
  );

describe('AssessmentsPaneIssuesSection', () => {
  test('a rerun over the same trace does not add a second card', () => {
    // Every detection run writes its own reference, so this used to show one card per run and an
    // issue count that climbed with each rerun.
    renderSection([
      reference({ assessment_id: 'a-first' }),
      reference({ assessment_id: 'a-second', last_update_time: '2026-10-06T05:35:58Z' }),
    ]);

    expect(screen.getAllByText('High latency')).toHaveLength(1);
    expect(screen.getByText(/Issues\s*\(1\)/)).toBeInTheDocument();
  });

  test('the newest reference is the one rendered', () => {
    renderSection([
      reference({ assessment_id: 'a-first', rationale: 'From the first run.' }),
      reference({
        assessment_id: 'a-second',
        last_update_time: '2026-10-06T05:35:58Z',
        rationale: 'From the rerun.',
      }),
    ]);

    expect(screen.getByTitle('High latency')).toBeInTheDocument();
    expect(screen.queryByText('From the first run.')).not.toBeInTheDocument();
  });

  test('distinct issues are each shown', () => {
    renderSection([
      reference({ assessment_id: 'a-latency' }),
      reference({ assessment_id: 'a-cost', assessment_name: 'iss-cost', issue: { issue_name: 'Runaway cost' } }),
    ]);

    expect(screen.getByText(/Issues\s*\(2\)/)).toBeInTheDocument();
  });

  test('an invalidated reference is not shown at all', () => {
    renderSection([reference({ assessment_id: 'a-invalid', valid: false })]);

    expect(screen.queryByText('High latency')).not.toBeInTheDocument();
  });
});
