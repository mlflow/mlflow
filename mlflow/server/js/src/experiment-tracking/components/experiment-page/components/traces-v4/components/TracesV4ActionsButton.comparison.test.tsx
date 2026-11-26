import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { TracesV4ActionsButton } from './TracesV4ActionsButton';
import type { TracesV4TraceActions } from '../hooks/useTracesV4TraceActions';
import { BASELINE_TRACE_ID_PARAM, CANDIDATE_TRACE_ID_PARAM } from '../hooks/useTracesV4UrlState';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { makeTrace } from '../test-utils/mockTraces';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/traces';
const URL = '/ml/experiments/exp-1/traces';

const { history } = setupTestRouter();

let lastSearch = '';
const LocationSpy = () => {
  const search = useLocation().search;
  useEffect(() => {
    lastSearch = search;
  }, [search]);
  return null;
};

const actions = {
  getTrace: jest.fn(),
  AddToReviewQueueDropdown: () => null,
  editTags: { showEditTagsModalForTrace: jest.fn(), EditTagsModal: null },
} as unknown as TracesV4TraceActions;

const renderButton = (selectedTraceInfos: ModelTraceInfoV3[]) =>
  renderTracesV4Page({
    initialUrl: URL,
    routes: [
      {
        path: PATH,
        element: (
          <>
            <LocationSpy />
            <TracesV4ActionsButton
              selectionCount={selectedTraceInfos.length}
              onDelete={() => {}}
              experimentId={EXPERIMENT_ID}
              actions={actions}
              selectedTraceInfos={selectedTraceInfos}
            />
          </>
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });

const openMenu = async () => {
  await userEvent.click(screen.getByRole('button', { name: 'Actions for selected traces' }));
};

beforeEach(() => {
  lastSearch = '';
  jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
});

describe('TracesV4ActionsButton trace comparison', () => {
  test('compares exactly two traces in the trace drawer with the new eval runs UI', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderButton([makeTrace('tr-000'), makeTrace('tr-001')]);

    await openMenu();
    expect(screen.queryByRole('menuitem', { name: 'Compare' })).not.toBeInTheDocument();
    expect(screen.getByText('Debug')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('menuitem', { name: 'Compare traces' }));

    await waitFor(() => {
      const params = new URLSearchParams(lastSearch);
      expect(params.get(BASELINE_TRACE_ID_PARAM)).toBe('trace:/cat.sch/tr-000');
      expect(params.get(CANDIDATE_TRACE_ID_PARAM)).toBe('trace:/cat.sch/tr-001');
      expect(params.get('traceId')).toBe('trace:/cat.sch/tr-001');
    });
  });

  test('offers no trace comparison for three selected traces with the new eval runs UI', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderButton([makeTrace('tr-000'), makeTrace('tr-001'), makeTrace('tr-002')]);

    await openMenu();
    expect(screen.queryByRole('menuitem', { name: 'Compare traces' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Compare' })).not.toBeInTheDocument();
  });

  test('keeps the comparison modal entry without the new eval runs UI', async () => {
    renderButton([makeTrace('tr-000'), makeTrace('tr-001')]);

    await openMenu();
    expect(screen.getByRole('menuitem', { name: 'Compare' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Compare traces' })).not.toBeInTheDocument();
    expect(screen.queryByText('Debug')).not.toBeInTheDocument();
  });
});
