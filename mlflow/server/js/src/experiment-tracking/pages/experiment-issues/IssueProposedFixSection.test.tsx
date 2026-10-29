import { beforeEach, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';

import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import type { Issue } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { IssueProposedFixSection } from './IssueProposedFixSection';

const ISSUE_ID = 'iss-a';

const issue: Issue = {
  issue_id: ISSUE_ID,
  experiment_id: '1',
  name: 'Refund falsely confirmed',
  description: 'The agent reports a refund the tool never completed.',
  severity: 'high',
  status: 'pending',
  created_timestamp: 1,
  last_updated_timestamp: 1,
  categories: ['correctness'],
};

type TraceSearchOutcome = 'error' | { traces: { trace_id: string }[] };

let respond: () => TraceSearchOutcome;

const server = setupServer(
  rest.post('*/ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
    const outcome = respond();
    return outcome === 'error' ? res(ctx.status(500), ctx.json({ message: 'nope' })) : res(ctx.json(outcome));
  }),
);

const renderSection = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  return renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <IssueProposedFixSection experimentId="1" issue={issue} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

beforeEach(() => {
  respond = () => ({ traces: [{ trace_id: 'tr-1' }, { trace_id: 'tr-2' }] });
});

describe('IssueProposedFixSection', () => {
  test('names the issue and its traces once they load', async () => {
    renderSection();

    await waitFor(() => expect(screen.getByRole('button', { name: 'Copy' })).toBeEnabled());
    const prompt = screen.getByTestId('experiment-issue-proposed-fix').querySelector('pre')?.textContent ?? '';
    expect(prompt).toContain('Use the mlflow-agent-improvement skill.');
    expect(prompt).toContain('Issue: Refund falsely confirmed');
    expect(prompt).toContain('- tr-1');
  });

  test('withholds Copy until the trace lookup settles', async () => {
    // An unsettled lookup reads as "no traces", and a prompt with none gives the agent nothing
    // to inspect, so copying waits rather than handing over a misleading prompt. Asserted at
    // mount, before the response lands, so no request has to be held open to observe it.
    renderSection();

    expect(screen.getByRole('button', { name: 'Copy' })).toBeDisabled();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Copy' })).toBeEnabled());
  });

  test('withholds Copy when the trace lookup failed', async () => {
    respond = () => 'error';
    renderSection();

    await waitFor(() => expect(screen.getByRole('button', { name: 'Copy' })).toBeDisabled());
  });

  test('allows Copy for an issue whose traces are genuinely gone', async () => {
    // A lookup that resolved to nothing is not a failure, and that prompt is honest.
    respond = () => ({ traces: [] });
    renderSection();

    await waitFor(() => expect(screen.getByRole('button', { name: 'Copy' })).toBeEnabled());
  });

  test('resolves the issue URL against the document base, keeping a static prefix', async () => {
    const base = document.createElement('base');
    base.href = `${window.location.origin}/mlflow/`;
    document.head.appendChild(base);
    try {
      renderSection();

      await waitFor(() => expect(screen.getByRole('button', { name: 'Copy' })).toBeEnabled());
      const prompt = screen.getByTestId('experiment-issue-proposed-fix').querySelector('pre')?.textContent ?? '';
      expect(prompt).toContain(`${window.location.origin}/mlflow/#`);
    } finally {
      document.head.removeChild(base);
    }
  });
});
