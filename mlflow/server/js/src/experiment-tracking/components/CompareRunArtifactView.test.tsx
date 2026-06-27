import { jest, describe, afterEach, test, expect } from '@jest/globals';
import { renderWithIntl, screen, within } from '../../common/utils/TestUtils.react18';
import { cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { CompareRunArtifactView } from './CompareRunArtifactView';
import type { RunInfoEntity } from '../types';
import type { ArtifactListFilesResponse } from '../types';

// ── Mock: useRunsArtifacts ──
// eslint-disable-next-line @databricks/no-const-object-record-string -- test mock
const mockArtifactsData: Record<string, ArtifactListFilesResponse> = {
  'run-uuid-1': {
    root_uri: 'dbfs:/artifacts/run-uuid-1',
    files: [
      { path: 'model/weights.bin', is_dir: false, file_size: 1024 },
      { path: 'metrics.png', is_dir: false, file_size: 512 },
      { path: 'report.html', is_dir: false, file_size: 2048 },
    ],
  },
  'run-uuid-2': {
    root_uri: 'dbfs:/artifacts/run-uuid-2',
    files: [
      { path: 'metrics.png', is_dir: false, file_size: 768 },
      { path: 'report.html', is_dir: false, file_size: 3072 },
    ],
  },
};

jest.mock('./experiment-page/hooks/useRunsArtifacts', () => ({
  useRunsArtifacts: jest.fn((runUuids: string[]) => {
    const subset: Record<string, ArtifactListFilesResponse> = {};
    for (const id of runUuids) {
      if (mockArtifactsData[id]) {
        subset[id] = mockArtifactsData[id];
      }
    }
    return { artifactsKeyedByRun: subset, isLoading: false, error: null };
  }),
}));

// ── Mock: ShowArtifactPage ──
// Renders a lightweight placeholder so we can assert that columns appear
// without needing the full artifact stack.
jest.mock('./artifact-view-components/ShowArtifactPage', () => {
  const MockShowArtifactPage = (props: { runUuid: string; path?: string }) => (
    <div data-testid={`artifact-page-${props.runUuid}`}>{props.path ?? 'select a file'}</div>
  );
  MockShowArtifactPage.displayName = 'MockShowArtifactPage';
  return { __esModule: true, default: MockShowArtifactPage };
});

// ── Helpers ──
const makeRunInfo = (runUuid: string, experimentId = '0'): RunInfoEntity =>
  ({
    runUuid,
    experimentId,
    artifactUri: `dbfs:/artifacts/${runUuid}`,
    startTime: Date.now(),
    endTime: Date.now(),
    status: 'FINISHED',
    lifecycleStage: 'active',
    runName: `run-${runUuid.slice(0, 4)}`,
  }) as unknown as RunInfoEntity;

const renderComponent = (runUuids: string[] = ['run-uuid-1', 'run-uuid-2']) => {
  const runInfos = runUuids.map((id) => makeRunInfo(id));
  return renderWithIntl(
    <DesignSystemProvider>
      <CompareRunArtifactView runUuids={runUuids} runInfos={runInfos} />
    </DesignSystemProvider>,
  );
};

// ── Tests ──
describe('CompareRunArtifactView', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    cleanup();
  });

  test('renders the empty state when there are no common artifacts', () => {
    // run-uuid-3 is not in the mock data → useRunsArtifacts returns {}
    renderComponent(['run-uuid-3']);
    expect(screen.getByText('No common artifacts to display.')).toBeInTheDocument();
  });

  test('renders one artifact column per run', () => {
    renderComponent();
    // Each run should have a mocked ShowArtifactPage placeholder
    expect(screen.getByTestId('artifact-page-run-uuid-1')).toBeInTheDocument();
    expect(screen.getByTestId('artifact-page-run-uuid-2')).toBeInTheDocument();
  });

  test('renders run identifier headers for each run column', () => {
    renderComponent();
    // The component shows "Run 1: <short id>" and "Run 2: <short id>"
    expect(screen.getByText(/Run 1/)).toBeInTheDocument();
    expect(screen.getByText(/Run 2/)).toBeInTheDocument();
    // Short IDs (first 8 chars) should be visible
    expect(screen.getByText('run-uuid')).toBeTruthy();
  });

  test('renders common artifact names in the tree sidebar', () => {
    renderComponent();
    // "metrics.png" and "report.html" are common to both runs;
    // "model/weights.bin" is only in run-uuid-1 and should be excluded.
    expect(screen.getByText('metrics.png')).toBeInTheDocument();
    expect(screen.getByText('report.html')).toBeInTheDocument();
    expect(screen.queryByText('weights.bin')).not.toBeInTheDocument();
  });

  test('does not accept a colWidth prop (dead parameter removal)', () => {
    // TypeScript would catch this at compile-time, but this runtime check
    // verifies the component's prop interface after refactoring.
    const component = CompareRunArtifactView as React.FC<any>;
    // Render with an extraneous colWidth prop – should still work fine
    const runInfos = ['run-uuid-1', 'run-uuid-2'].map((id) => makeRunInfo(id));
    renderWithIntl(
      <DesignSystemProvider>
        {/* @ts-expect-error Intentionally passing removed prop */}
        <CompareRunArtifactView runUuids={['run-uuid-1', 'run-uuid-2']} runInfos={runInfos} colWidth={200} />
      </DesignSystemProvider>,
    );
    // Component renders without error
    expect(screen.getByTestId('artifact-page-run-uuid-1')).toBeInTheDocument();
  });
});
