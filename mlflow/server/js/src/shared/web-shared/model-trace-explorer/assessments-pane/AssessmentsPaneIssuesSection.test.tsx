import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { MemoryRouter, Route, Routes } from 'react-router-dom';

import { searchIssueIdsByStatus } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery';

import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';
import { AssessmentsPaneIssuesSection } from './AssessmentsPaneIssuesSection';
import type { IssueReferenceAssessment } from '../ModelTrace.types';

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery', () => ({
  ...jest.requireActual<
    typeof import('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery')
  >('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery'),
  searchIssueIdsByStatus: jest.fn(),
}));

const mockSearchIssueIdsByStatus = jest.mocked(searchIssueIdsByStatus);

/** Answers the experiment's dismissed-issue lookup. */
const givenDismissedIssueIds = (issueIds: string[]) => mockSearchIssueIdsByStatus.mockResolvedValue(issueIds);

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
        {/* The section looks up the experiment's dismissed issues, which it reads off the route. */}
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <MemoryRouter initialEntries={['/experiments/experiment-1/traces']}>
            <Routes>
              <Route path="/experiments/:experimentId/*" element={<AssessmentsPaneIssuesSection issues={issues} />} />
            </Routes>
          </MemoryRouter>
        </QueryClientProvider>
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

  describe('issues dismissed as not an issue', () => {
    beforeEach(() => {
      mockSearchIssueIdsByStatus.mockReset();
    });

    test('a rejected issue is dropped from the list and its count', async () => {
      givenDismissedIssueIds(['iss-latency']);
      renderSection([
        reference({ assessment_id: 'a-latency' }),
        reference({ assessment_id: 'a-cost', assessment_name: 'iss-cost', issue: { issue_name: 'Runaway cost' } }),
      ]);

      // Shown until the lookup answers, so the dismissed one is waited out.
      await waitFor(() => expect(screen.queryByText('High latency')).not.toBeInTheDocument());
      expect(screen.getByText('Runaway cost')).toBeInTheDocument();
      expect(screen.getByText(/Issues\s*\(1\)/)).toBeInTheDocument();
    });

    test('only dismissed issues are looked up, so a resolved one is kept', async () => {
      givenDismissedIssueIds([]);
      renderSection([reference({ assessment_id: 'a-latency' })]);

      await waitFor(() => expect(mockSearchIssueIdsByStatus).toHaveBeenCalledWith('experiment-1', 'rejected'));
      expect(screen.getByText('High latency')).toBeInTheDocument();
    });
  });
});
