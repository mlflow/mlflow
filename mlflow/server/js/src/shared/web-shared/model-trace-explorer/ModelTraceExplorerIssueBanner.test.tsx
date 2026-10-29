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
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { searchIssueIdsByStatus } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery';

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery', () => ({
  ...jest.requireActual<
    typeof import('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery')
  >('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery'),
  searchIssueIdsByStatus: jest.fn(),
}));

const mockSearchIssueIdsByStatus = jest.mocked(searchIssueIdsByStatus);

/** Answers the experiment's dismissed-issue lookup. */
const givenDismissedIssueIds = (issueIds: string[]) => mockSearchIssueIdsByStatus.mockResolvedValue(issueIds);

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

/**
 * The explorer is mounted under an experiment route, which is where the dismissed-issue lookup
 * reads the experiment from, and the query string carries the issue the reader followed here.
 */
const renderInProviders = (children: React.ReactNode, route = '/experiments/experiment-1/traces') =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <MemoryRouter initialEntries={[route]}>
            <Routes>
              <Route path="/experiments/:experimentId/*" element={children} />
            </Routes>
          </MemoryRouter>
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
    mockSearchIssueIdsByStatus.mockReset();
  });

  // Named apart from each other, so every assertion below says which issue it matched.
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

  const renderEvidence = (assessments: Assessment[], route?: string) =>
    renderInProviders(
      <ModelTraceExplorerIssueEvidence activeSpan={makeSpan(assessments)} experimentId="experiment-1" />,
      route,
    );

  it('drops a dismissed issue from the cards and keeps the rest', async () => {
    givenDismissedIssueIds(['iss-dismissed']);
    renderEvidence([dismissed, kept]);

    // Both show until the lookup answers, so the dismissed one is waited out.
    await waitFor(() => expect(screen.queryByText('Not really a bug')).not.toBeInTheDocument());
    expect(screen.getByText('Genuinely slow')).toBeInTheDocument();
  });

  it('asks only for the dismissed status, so a resolved issue is left alone', async () => {
    givenDismissedIssueIds([]);
    renderEvidence([kept]);

    await waitFor(() => expect(mockSearchIssueIdsByStatus).toHaveBeenCalledWith('experiment-1', 'rejected'));
    expect(screen.getByText('Genuinely slow')).toBeInTheDocument();
  });

  it('keeps a dismissed issue that the reader navigated to', async () => {
    // Both are dismissed, so the one that was not followed here disappearing is what proves the
    // lookup had already answered.
    givenDismissedIssueIds(['iss-dismissed', 'iss-kept']);
    renderEvidence([dismissed, kept], '/experiments/experiment-1/issues?selectedIssueId=iss-dismissed');

    await waitFor(() => expect(screen.queryByText('Genuinely slow')).not.toBeInTheDocument());
    expect(screen.getByText('Not really a bug')).toBeInTheDocument();
  });

  it('keeps every issue when the lookup fails', async () => {
    mockSearchIssueIdsByStatus.mockRejectedValue(new Error('issues API unavailable'));
    renderEvidence([dismissed]);

    await waitFor(() => expect(mockSearchIssueIdsByStatus).toHaveBeenCalled());
    expect(screen.getByText('Not really a bug')).toBeInTheDocument();
  });

  it('keeps a dismissed issue when it is not on an experiment route', async () => {
    givenDismissedIssueIds(['iss-dismissed']);
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
            <MemoryRouter initialEntries={['/']}>
              <ModelTraceExplorerIssueEvidence activeSpan={makeSpan([dismissed])} experimentId="experiment-1" />
            </MemoryRouter>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(await screen.findByText('Not really a bug')).toBeInTheDocument();
    expect(mockSearchIssueIdsByStatus).not.toHaveBeenCalled();
  });

  it('stops counting a dismissed issue on the span tree', async () => {
    givenDismissedIssueIds(['iss-dismissed']);
    renderInProviders(<ModelTraceExplorerIssueIndicator span={makeSpan([dismissed, kept])} />);

    // The tag falls back to naming the one issue left, which is how the survivor is identified.
    expect(screen.getByLabelText('2 issues detected')).toHaveTextContent('2');
    expect(await screen.findByLabelText('Issue detected: Genuinely slow')).toHaveTextContent('1');
  });
});
