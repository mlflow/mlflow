import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import {
  ModelSpanType,
  type Assessment,
  type IssueReferenceAssessment,
  type ModelTraceSpanNode,
} from './ModelTrace.types';
import { ModelTraceExplorerIssueEvidence, ModelTraceExplorerIssueIndicator } from './ModelTraceExplorerIssueBanner';
import { QueryClient, QueryClientProvider } from '../query-client/queryClient';
import { BrowserRouter } from './RoutingUtils';
import { MemoryRouter } from 'react-router-dom';
import { getIssue } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery';

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery', () => ({
  ...jest.requireActual<
    typeof import('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery')
  >('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery'),
  getIssue: jest.fn(),
}));

const mockGetIssue = jest.mocked(getIssue);

/** Answers the status read for each issue ID; an ID left out resolves as still open. */
const givenIssueStatuses = (statuses: Record<string, string>) =>
  mockGetIssue.mockImplementation((issueId: string) =>
    Promise.resolve({ issue_id: issueId, status: statuses[issueId] ?? 'pending' } as Awaited<
      ReturnType<typeof getIssue>
    >),
  );

const issueReference: IssueReferenceAssessment = {
  assessment_id: 'a-original',
  assessment_name: 'iss-latency',
  trace_id: 'trace-1',
  span_id: 'span-1',
  source: { source_type: 'LLM_JUDGE', source_id: 'mlflow.issue_detection' },
  create_time: '2026-10-05T05:35:58Z',
  last_update_time: '2026-10-05T05:35:58Z',
  issue: { issue_name: 'High latency' },
  rationale: 'The tool took too long.',
};

const ordinaryFeedback: Assessment = {
  assessment_id: 'a-feedback',
  assessment_name: 'quality',
  trace_id: 'trace-1',
  span_id: 'span-1',
  source: issueReference.source,
  create_time: issueReference.create_time,
  last_update_time: issueReference.last_update_time,
  rationale: issueReference.rationale,
  feedback: { value: true },
};

const emptyCases: { name: string; assessments: Assessment[] }[] = [
  { name: 'empty', assessments: [] },
  { name: 'ordinary feedback', assessments: [ordinaryFeedback] },
  { name: 'invalid references', assessments: [{ ...issueReference, valid: false }] },
];

const distinctIssues: Assessment[] = [
  issueReference,
  { ...issueReference, assessment_id: 'a-other', assessment_name: 'iss-other' },
  ordinaryFeedback,
  { ...issueReference, assessment_id: 'a-invalid', assessment_name: 'iss-invalid', valid: false },
];

const makeSpan = (assessments: Assessment[]): ModelTraceSpanNode => ({
  key: 'span-1',
  title: 'Tool call',
  type: ModelSpanType.LLM,
  start: 0,
  end: 1,
  inputs: {},
  outputs: {},
  attributes: {},
  assessments,
  traceId: 'trace-1',
});

const renderInProviders = (children: React.ReactNode) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        {/* The cards read each issue's status to hide the ones dismissed as not an issue. */}
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <BrowserRouter>{children}</BrowserRouter>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ModelTraceExplorerIssueEvidence', () => {
  // Passed explicitly rather than defaulted: one case renders without an experiment to link to.
  const renderEvidence = (assessments: Assessment[], experimentId?: string) =>
    renderInProviders(
      <ModelTraceExplorerIssueEvidence activeSpan={makeSpan(assessments)} experimentId={experimentId} />,
    );

  it('names the issue a span was flagged for and links to its detail pane', () => {
    renderEvidence([issueReference, { ...issueReference, assessment_id: 'a-repeated' }], 'experiment-1');

    const issueCard = within(screen.getByRole('status'));
    expect(issueCard.getByText('High latency')).toBeVisible();
    expect(issueCard.getByText('The tool took too long.')).toBeVisible();
    expect(issueCard.getByRole('link')).toHaveAttribute(
      'href',
      '/experiments/experiment-1/issues?selectedIssueId=iss-latency',
    );
  });

  it('shows the newest title and rationale when an older reference arrives last', () => {
    renderEvidence(
      [
        {
          ...issueReference,
          assessment_id: 'a-newest',
          last_update_time: '2026-10-06T00:00:00Z',
          issue: { issue_name: 'Updated issue title' },
          rationale: 'Updated evidence.',
        },
        issueReference,
      ],
      'experiment-1',
    );

    const issueCard = within(screen.getByRole('status'));
    expect(issueCard.getByText('Updated issue title')).toBeVisible();
    expect(issueCard.getByText('Updated evidence.')).toBeVisible();
  });

  it('preserves distinct issue IDs even when their titles match', () => {
    renderEvidence(distinctIssues, 'experiment-1');

    const issueHrefs = screen.getAllByRole('status').map((card) => {
      const issueCard = within(card);
      expect(issueCard.getByText('High latency')).toBeVisible();
      return issueCard.getByRole('link').getAttribute('href');
    });
    expect(issueHrefs.sort()).toEqual([
      '/experiments/experiment-1/issues?selectedIssueId=iss-latency',
      '/experiments/experiment-1/issues?selectedIssueId=iss-other',
    ]);
  });

  it('still names the issue where the host supplies no experiment to link to', () => {
    renderEvidence([issueReference]);

    const issueCard = within(screen.getByRole('status'));
    expect(issueCard.getByText('High latency')).toBeVisible();
    expect(issueCard.queryByRole('link')).not.toBeInTheDocument();
  });

  it.each(emptyCases)('shows no issue card for $name assessments', ({ assessments }) => {
    renderEvidence(assessments, 'experiment-1');

    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.queryByText('High latency')).not.toBeInTheDocument();
  });
});

describe('ModelTraceExplorerIssueIndicator', () => {
  const renderIndicator = (assessments: Assessment[]) =>
    renderInProviders(<ModelTraceExplorerIssueIndicator span={makeSpan(assessments)} />);

  it('counts repeated references to one issue once', () => {
    renderIndicator([issueReference, { ...issueReference, assessment_id: 'a-repeated' }]);

    expect(screen.getByLabelText('Issue detected: High latency')).toHaveTextContent('1');
  });

  it('counts distinct issue IDs with matching titles separately', () => {
    renderIndicator(distinctIssues);

    expect(screen.getByLabelText('2 issues detected')).toHaveTextContent('2');
  });

  it.each(emptyCases)('shows no issue indicator for $name assessments', ({ assessments }) => {
    renderIndicator(assessments);

    expect(screen.queryByLabelText('Issue detected: High latency')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('2 issues detected')).not.toBeInTheDocument();
  });
});

describe('issues dismissed as not an issue', () => {
  beforeEach(() => {
    mockGetIssue.mockReset();
  });

  // Named apart from each other, so every assertion below says which issue it matched. The
  // same-title fixtures above would let a card or a label survive either filtering decision.
  const dismissed: Assessment = {
    ...issueReference,
    assessment_name: 'iss-dismissed',
    issue: { issue_name: 'Not really a bug' },
  };
  const kept: Assessment = {
    ...issueReference,
    assessment_id: 'a-kept',
    assessment_name: 'iss-kept',
    issue: { issue_name: 'Genuinely slow' },
  };

  const renderEvidence = (assessments: Assessment[]) =>
    renderInProviders(
      <ModelTraceExplorerIssueEvidence activeSpan={makeSpan(assessments)} experimentId="experiment-1" />,
    );

  it('drops a rejected issue from the cards and keeps a resolved one', async () => {
    givenIssueStatuses({ 'iss-dismissed': 'rejected', 'iss-kept': 'resolved' });
    renderEvidence([dismissed, kept]);

    // Both show until their statuses arrive, so the dismissed one has to be waited out.
    await waitFor(() => expect(screen.queryByText('Not really a bug')).not.toBeInTheDocument());
    // Resolved is left alone: that issue did happen in this trace.
    expect(screen.getByText('Genuinely slow')).toBeInTheDocument();
  });

  it('keeps a rejected issue that the reader navigated to', async () => {
    // Arriving from the issue's own page is how its evidence gets reviewed, so the card it was
    // followed from must survive even though it is dismissed.
    givenIssueStatuses({ 'iss-latency': 'rejected' });
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
            <MemoryRouter initialEntries={['/?selectedIssueId=iss-latency']}>
              <ModelTraceExplorerIssueEvidence activeSpan={makeSpan([issueReference])} experimentId="experiment-1" />
            </MemoryRouter>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(await screen.findByText('High latency')).toBeInTheDocument();
    // Never read, because the answer could not change the outcome.
    expect(mockGetIssue).not.toHaveBeenCalled();
  });

  it('keeps an issue whose status cannot be read', async () => {
    mockGetIssue.mockRejectedValue(new Error('issues API unavailable'));
    renderEvidence([issueReference]);

    await waitFor(() => expect(mockGetIssue).toHaveBeenCalled());
    expect(screen.getByText('High latency')).toBeInTheDocument();
  });

  it('stops counting a rejected issue on the span tree', async () => {
    givenIssueStatuses({ 'iss-dismissed': 'rejected', 'iss-kept': 'pending' });
    renderInProviders(<ModelTraceExplorerIssueIndicator span={makeSpan([dismissed, kept])} />);

    // Both are counted until the statuses arrive, then the dismissed one drops out and the tag
    // falls back to naming the one issue left -- by name, so it is the survivor being asserted on.
    expect(screen.getByLabelText('2 issues detected')).toHaveTextContent('2');
    expect(await screen.findByLabelText('Issue detected: Genuinely slow')).toHaveTextContent('1');
  });
});
