import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { rest } from 'msw';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor, within } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { ISSUE_EVIDENCE_METADATA_KEY } from '../../../shared/web-shared/model-trace-explorer';
import { IssueEvidenceSection } from './IssueEvidenceSection';

jest.mock('./IssueTraceDrawer', () => ({
  IssueTraceDrawer: () => <div data-testid="issue-trace-drawer" />,
}));

const ISSUE_ID = 'iss-abc';

const evidenceMetadata = (steps: unknown[]) => JSON.stringify({ schema_version: 1, steps });

const issueReference = ({
  spanId,
  rationale = 'The agent answered from an empty context',
  metadata,
  valid,
  assessmentId = 'a-1',
  lastUpdateTime = '2026-10-16T09:00:00.000Z',
}: {
  spanId?: string;
  rationale?: string;
  metadata?: Record<string, string>;
  valid?: boolean;
  assessmentId?: string;
  lastUpdateTime?: string;
}) => ({
  assessment_id: assessmentId,
  assessment_name: ISSUE_ID,
  issue: { issue_name: 'Empty retrieval' },
  span_id: spanId,
  rationale,
  metadata,
  valid,
  create_time: '2026-10-16T09:00:00.000Z',
  last_update_time: lastUpdateTime,
});

const traceInfo = (traceId: string, assessments: unknown[] = [], overrides: Record<string, unknown> = {}) => ({
  trace_id: traceId,
  trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: '1' } },
  request_time: '2026-10-16T09:00:00.000Z',
  execution_duration: '1.4s',
  state: 'OK',
  tags: { 'mlflow.traceName': `trace_${traceId}` },
  assessments,
  ...overrides,
});

const span = (spanId: string, overrides: Record<string, unknown> = {}) => ({
  trace_id: 'tr-1',
  span_id: spanId,
  parent_span_id: null,
  name: `span_${spanId}`,
  trace_state: '',
  start_time_unix_nano: '0',
  end_time_unix_nano: '1',
  status: { code: 'STATUS_CODE_OK' },
  attributes: {},
  ...overrides,
});

let searchBodies: Record<string, unknown>[];
let tracePages: { traces: unknown[]; next_page_token?: string }[];
let spansByTraceId: Record<string, unknown[]>;
let spanRequests: string[];

const server = setupServer(
  rest.post('*/ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
    const body = await req.json();
    searchBodies.push(body);
    const page = tracePages[searchBodies.length - 1] ?? { traces: [] };
    return res(ctx.json(page));
  }),
  rest.get('*/ajax-api/3.0/mlflow/traces/:traceId', (req, res, ctx) =>
    res(ctx.json({ trace: { trace_info: traceInfo(String(req.params['traceId'])) } })),
  ),
  rest.get('*/ajax-api/3.0/mlflow/get-trace-artifact', (req, res, ctx) => {
    const traceId = String(req.url.searchParams.get('request_id'));
    spanRequests.push(traceId);
    return res(ctx.json({ spans: spansByTraceId[traceId] ?? [] }));
  }),
);

const renderSection = (issueId = ISSUE_ID) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  const onViewTrace = jest.fn();
  const result = renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <IssueEvidenceSection experimentId="1" issueId={issueId} onViewTrace={onViewTrace} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { ...result, onViewTrace };
};

beforeEach(() => {
  searchBodies = [];
  tracePages = [];
  spansByTraceId = {};
  spanRequests = [];
});

describe('IssueEvidenceSection', () => {
  test('finds affected traces through the issue reference filter', async () => {
    tracePages = [{ traces: [traceInfo('tr-1', [issueReference({})])] }];
    renderSection();

    expect(await screen.findByRole('button', { name: /trace_tr-1/ })).toBeInTheDocument();
    expect(searchBodies[0]).toEqual(
      expect.objectContaining({ filter: `issue.id = '${ISSUE_ID}'`, order_by: ['timestamp DESC'] }),
    );
  });

  test('renders the stored walkthrough with its highlights and summary', async () => {
    tracePages = [
      {
        traces: [
          traceInfo('tr-1', [
            issueReference({
              spanId: 'span-2',
              metadata: {
                [ISSUE_EVIDENCE_METADATA_KEY]: evidenceMetadata([
                  {
                    span_id: 'span-1',
                    text: 'The retriever returned 0 documents',
                    highlights: ['0 documents'],
                    highlight_tone: 'error',
                  },
                  {
                    span_id: 'span-2',
                    text: 'The model answered anyway',
                    highlights: ['anyway'],
                    highlight_tone: 'warning',
                  },
                ]),
              },
            }),
          ]),
        ],
      },
    ];
    spansByTraceId['tr-1'] = [span('span-1'), span('span-2')];
    renderSection();

    expect(await screen.findByText('The agent answered from an empty context')).toBeInTheDocument();
    // Highlights are rendered as emphasis inside the step's own text.
    const step = await screen.findByText(/The retriever returned/);
    expect(within(step).getByText('0 documents')).toBeInTheDocument();
    // Each step is attributed to the span it cites, by that span's name.
    expect(await screen.findByText('span_span-1')).toBeInTheDocument();
    expect(screen.getByText('span_span-2')).toBeInTheDocument();
  });

  test('ignores evidence whose highlight is not in its text', async () => {
    tracePages = [
      {
        traces: [
          traceInfo('tr-1', [
            issueReference({
              metadata: {
                [ISSUE_EVIDENCE_METADATA_KEY]: evidenceMetadata([
                  {
                    span_id: 'span-1',
                    text: 'The retriever returned 0 documents',
                    highlights: ['a paraphrase'],
                    highlight_tone: 'error',
                  },
                ]),
              },
            }),
          ]),
        ],
      },
    ];
    renderSection();

    expect(await screen.findByRole('button', { name: /trace_tr-1/ })).toBeInTheDocument();
    expect(screen.queryByText(/The retriever returned/)).not.toBeInTheDocument();
    // A reference with unreadable evidence still shows what it does say.
    expect(screen.getByText('The agent answered from an empty context')).toBeInTheDocument();
  });

  test.each([
    ['malformed JSON', '{not json'],
    ['another schema version', JSON.stringify({ schema_version: 2, steps: [] })],
  ])('falls back to the trace itself for evidence with %s', async (_label, metadataValue) => {
    tracePages = [
      {
        traces: [
          traceInfo('tr-1', [
            issueReference({
              spanId: 'span-1',
              metadata: { [ISSUE_EVIDENCE_METADATA_KEY]: metadataValue },
            }),
          ]),
        ],
      },
    ];
    spansByTraceId['tr-1'] = [
      span('span-1', {
        status: { code: 'STATUS_CODE_ERROR' },
        events: [{ name: 'exception', attributes: { 'exception.message': 'upstream timed out' } }],
      }),
    ];
    renderSection();

    expect(await screen.findByText(/upstream timed out/)).toBeInTheDocument();
  });

  test('describes the blamed span for a reference that carries only a rationale', async () => {
    tracePages = [{ traces: [traceInfo('tr-1', [issueReference({ spanId: 'span-1' })])] }];
    spansByTraceId['tr-1'] = [
      span('span-1', { attributes: { 'mlflow.spanInputs': JSON.stringify({ query: 'who signed it?' }) } }),
    ];
    renderSection();

    expect(await screen.findByText(/who signed it\?/)).toBeInTheDocument();
  });

  test('reads no spans for a reference that names none', async () => {
    tracePages = [{ traces: [traceInfo('tr-1', [issueReference({})])] }];
    renderSection();

    expect(await screen.findByRole('button', { name: /trace_tr-1/ })).toBeInTheDocument();
    // Its rationale is the whole explanation, and explaining it needs no span read.
    await waitFor(() => expect(screen.getByText('The agent answered from an empty context')).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: 'View span' })).not.toBeInTheDocument();
    expect(spanRequests).toEqual([]);
  });

  test('reads spans only for the cards that are open', async () => {
    tracePages = [
      {
        traces: [
          traceInfo('tr-1', [issueReference({ spanId: 'span-1' })]),
          traceInfo('tr-2', [issueReference({ spanId: 'span-1', assessmentId: 'a-2' })]),
        ],
      },
    ];
    spansByTraceId['tr-1'] = [span('span-1')];
    spansByTraceId['tr-2'] = [span('span-1')];
    renderSection();

    // The leading card opens by default; the second stays collapsed and costs no span read.
    await waitFor(() => expect(spanRequests).toEqual(['tr-1']));

    await userEvent.click(screen.getByRole('button', { name: 'Expand evidence for trace_tr-2' }));
    await waitFor(() => expect(spanRequests).toEqual(['tr-1', 'tr-2']));
  });

  test('collapsing the open card does not reopen it on the next toggle', async () => {
    tracePages = [
      {
        traces: [traceInfo('tr-1', [issueReference({})]), traceInfo('tr-2', [issueReference({ assessmentId: 'a-2' })])],
      },
    ];
    renderSection();

    await userEvent.click(await screen.findByRole('button', { name: 'Collapse evidence for trace_tr-1' }));
    await userEvent.click(screen.getByRole('button', { name: 'Expand evidence for trace_tr-2' }));

    expect(screen.getByRole('button', { name: 'Expand evidence for trace_tr-1' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Collapse evidence for trace_tr-2' })).toBeInTheDocument();
  });

  test('opens a trace on the span its evidence blames', async () => {
    tracePages = [
      {
        traces: [
          traceInfo('tr-1', [
            issueReference({
              spanId: 'span-2',
              metadata: {
                [ISSUE_EVIDENCE_METADATA_KEY]: evidenceMetadata([
                  {
                    span_id: 'span-1',
                    text: 'The retriever returned 0 documents',
                    highlights: ['0 documents'],
                    highlight_tone: 'error',
                  },
                ]),
              },
            }),
          ]),
        ],
      },
    ];
    spansByTraceId['tr-1'] = [span('span-1'), span('span-2')];
    const { onViewTrace } = renderSection();

    await userEvent.click(await screen.findByRole('button', { name: 'Open trace' }));
    expect(onViewTrace).toHaveBeenCalledWith('tr-1', 'span-2');

    await userEvent.click(screen.getByRole('button', { name: 'View span' }));
    expect(onViewTrace).toHaveBeenLastCalledWith('tr-1', 'span-1');
  });

  test('prefers the newest valid reference when a trace carries several', async () => {
    tracePages = [
      {
        traces: [
          traceInfo('tr-1', [
            issueReference({
              assessmentId: 'a-old',
              lastUpdateTime: '2026-10-15T09:00:00.000Z',
              rationale: 'An earlier detection run',
            }),
            issueReference({
              assessmentId: 'a-new',
              lastUpdateTime: '2026-10-16T09:00:00.000Z',
              rationale: 'The latest detection run',
            }),
            issueReference({
              assessmentId: 'a-invalid',
              lastUpdateTime: '2026-10-17T09:00:00.000Z',
              rationale: 'An invalidated reference',
              valid: false,
            }),
          ]),
        ],
      },
    ];
    renderSection();

    expect(await screen.findByText('The latest detection run')).toBeInTheDocument();
    expect(screen.queryByText('An earlier detection run')).not.toBeInTheDocument();
    expect(screen.queryByText('An invalidated reference')).not.toBeInTheDocument();
  });

  test('ignores a reference to a different issue on the same trace', async () => {
    const otherIssue = { ...issueReference({ rationale: 'Another issue entirely' }), assessment_name: 'iss-other' };
    tracePages = [{ traces: [traceInfo('tr-1', [otherIssue])] }];
    renderSection();

    expect(await screen.findByRole('button', { name: /trace_tr-1/ })).toBeInTheDocument();
    expect(screen.queryByText('Another issue entirely')).not.toBeInTheDocument();
  });

  test('pages through the remaining affected traces', async () => {
    tracePages = [
      { traces: [traceInfo('tr-1', [issueReference({})])], next_page_token: 'page-2' },
      { traces: [traceInfo('tr-2', [issueReference({ assessmentId: 'a-2' })])] },
    ];
    renderSection();

    await userEvent.click(await screen.findByRole('button', { name: /Show more/ }));

    expect(await screen.findByRole('button', { name: /trace_tr-2/ })).toBeInTheDocument();
    expect(searchBodies[1]).toEqual(expect.objectContaining({ page_token: 'page-2' }));
    expect(screen.queryByRole('button', { name: /Show more/ })).not.toBeInTheDocument();
  });

  test('distinguishes an issue with no affected traces from a failed search', async () => {
    tracePages = [{ traces: [] }];
    const { unmount } = renderSection();
    expect(await screen.findByText('No affected traces are recorded for this issue.')).toBeInTheDocument();
    unmount();

    server.use(rest.post('*/ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => res(ctx.status(500))));
    renderSection();
    expect(await screen.findByText('The traces behind this issue could not be loaded.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  test('does not search with an issue id that could not be one', async () => {
    renderSection("iss-' or '1'='1");

    await waitFor(() =>
      expect(screen.getByText('No affected traces are recorded for this issue.')).toBeInTheDocument(),
    );
    expect(searchBodies).toEqual([]);
  });
});
