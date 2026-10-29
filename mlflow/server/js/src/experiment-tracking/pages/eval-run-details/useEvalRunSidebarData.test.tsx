import { describe, expect, jest, test } from '@jest/globals';
import type { ReactNode } from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { searchMlflowTracesQueryFn } from '@databricks/web-shared/genai-traces-table';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { useGetRunQuery } from '../../components/run-page/hooks/useGetRunQuery';
import { useEvalRunSidebarData } from './useEvalRunSidebarData';

jest.mock('../../components/run-page/hooks/useGetRunQuery', () => ({ useGetRunQuery: jest.fn() }));
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  ),
  searchMlflowTracesQueryFn: jest.fn(),
}));

const RUN_UUID = 'run-1';

const feedback = (name: string, metadata: Record<string, string> = {}) => ({
  assessment_id: `${name}-${Math.random()}`,
  assessment_name: name,
  trace_id: 'tr',
  source: { source_type: 'LLM_JUDGE', source_id: name },
  create_time: '2026-01-01T00:00:00Z',
  last_update_time: '2026-01-01T00:00:00Z',
  feedback: { value: 'yes' },
  metadata: { 'mlflow.assessment.sourceRunId': RUN_UUID, ...metadata },
});

const trace = (traceId: string, assessments: ReturnType<typeof feedback>[]) =>
  ({
    trace_id: traceId,
    request_time: '2026-01-01T00:00:00Z',
    state: 'OK',
    trace_metadata: { 'mlflow.sourceRun': RUN_UUID },
    tags: {},
    assessments,
  }) as unknown as ModelTraceInfoV3;

const renderSidebarHook = (
  runTags: { key: string; value: string }[],
  traces: ModelTraceInfoV3[],
  modelInputs: { modelId: string }[] = [],
) => {
  jest.mocked(useGetRunQuery).mockReturnValue({
    data: {
      info: { userId: 'me', startTime: '0' },
      data: { tags: runTags, params: [] },
      inputs: { datasetInputs: [], modelInputs },
    },
    loading: false,
  } as unknown as ReturnType<typeof useGetRunQuery>);
  jest.mocked(searchMlflowTracesQueryFn).mockResolvedValue(traces);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <IntlProvider locale="en">
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </IntlProvider>
  );
  return renderHook(() => useEvalRunSidebarData({ experimentId: 'exp-1', runUuid: RUN_UUID }), { wrapper });
};

describe('useEvalRunSidebarData lineage', () => {
  test('reads the linked prompt versions from the run tag and keeps mlflow tags out of the tag list', async () => {
    const { result } = renderSidebarHook(
      [
        { key: 'mlflow.linkedPrompts', value: JSON.stringify([{ name: 'greeting prompt', version: '3' }]) },
        { key: 'team', value: 'a' },
      ],
      [],
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.sidebarProps.prompts).toEqual([{ name: 'greeting prompt', version: '3' }]);
    expect(result.current.sidebarProps.tags).toEqual([{ key: 'team', value: 'a' }]);
  });

  test('links a scorer only when every assessment with its name came from the same registered scorer', async () => {
    const registered = { 'mlflow.assessment.scorerName': 'quality', 'mlflow.assessment.scorerVersion': '2' };
    const { result } = renderSidebarHook(
      [],
      [
        trace('tr-1', [feedback('quality', registered), feedback('tone')]),
        trace('tr-2', [feedback('quality', registered), feedback('tone')]),
      ],
    );

    await waitFor(() => expect(result.current.sidebarProps.scorers).toHaveLength(2));
    const scorersByName = Object.fromEntries(
      result.current.sidebarProps.scorers.map((scorer) => [scorer.name, scorer.scorerReference]),
    );
    expect(scorersByName['quality']).toEqual({ scorerName: 'quality', scorerVersion: 2 });
    expect(scorersByName['tone']).toBeUndefined();
  });
});

describe('useEvalRunSidebarData evaluation lineage', () => {
  test('reads the agent, git commit and scorers digest from the run, and the logged model from its inputs', async () => {
    const { result } = renderSidebarHook(
      [
        { key: 'mlflow.genaiEvaluate.agent.function', value: 'my_pkg.agent.predict' },
        { key: 'mlflow.genaiEvaluate.agent.digest', value: 'sha256:agent' },
        {
          key: 'mlflow.genaiEvaluate.scorersDigest',
          value: JSON.stringify({ digest: 'sha256:s', hashed: 1, total: 2 }),
        },
        { key: 'mlflow.source.git.commit', value: '0123456789abcdef' },
      ],
      [],
      [{ modelId: 'm-123' }],
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.sidebarProps.agent).toEqual({
      function: 'my_pkg.agent.predict',
      uri: undefined,
      digest: 'sha256:agent',
      loggedModelId: 'm-123',
    });
    expect(result.current.sidebarProps.gitCommit).toBe('0123456789abcdef');
    expect(result.current.sidebarProps.scorersDigest).toEqual({
      digest: 'sha256:s',
      hashed: 1,
      total: 2,
      coverage: 'partial',
    });
  });

  test('has no lineage for a run without the tags', async () => {
    const { result } = renderSidebarHook([{ key: 'team', value: 'a' }], []);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.sidebarProps.agent).toBeUndefined();
    expect(result.current.sidebarProps.gitCommit).toBeUndefined();
    expect(result.current.sidebarProps.scorersDigest).toBeUndefined();
  });

  test('gives each scorer the definition digest its feedback reports', async () => {
    const { result } = renderSidebarHook(
      [],
      [
        trace('tr-1', [feedback('quality', { 'mlflow.assessment.scorerDigest': 'sha256:q' }), feedback('tone')]),
        trace('tr-2', [feedback('quality', { 'mlflow.assessment.scorerDigest': 'sha256:q' })]),
      ],
    );

    await waitFor(() => expect(result.current.sidebarProps.scorers).toHaveLength(2));
    const digestsByName = Object.fromEntries(
      result.current.sidebarProps.scorers.map((scorer) => [scorer.name, scorer.digest]),
    );
    expect(digestsByName['quality']).toBe('sha256:q');
    expect(digestsByName['tone']).toBeUndefined();
  });
});
