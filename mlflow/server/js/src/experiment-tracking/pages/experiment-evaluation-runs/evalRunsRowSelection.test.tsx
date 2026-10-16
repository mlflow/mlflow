import { jest, describe, test, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { RowSelectionState } from '@tanstack/react-table';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import { getEvalRunsBaseSelectionState } from './ExperimentEvaluationRunsTable.constants';
import { ExperimentEvaluationRunsRowVisibilityProvider } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import type { RunEntityWithChildren } from '../../types';
import { TestRouter, testRoute } from '../../../common/utils/RoutingTestUtils';

jest.mock('../../components/experiment-page/hooks/useExperimentRunColor', () => ({
  useGetExperimentRunColor: () => () => '#000000',
  useSaveExperimentRunColor: () => jest.fn(),
}));

const RUN_IDS = ['run-1', 'run-2', 'run-3', 'run-4'];

const makeRun = (runUuid: string): RunEntityWithChildren =>
  ({
    data: { params: [], tags: [], metrics: [] },
    info: {
      artifactUri: '',
      endTime: 0,
      experimentId: 'exp-1',
      lifecycleStage: '',
      runUuid,
      runName: `Run ${runUuid}`,
      startTime: 0,
      status: 'FINISHED',
    },
    inputs: { datasetInputs: [], modelInputs: [] },
    outputs: { modelOutputs: [] },
  }) as unknown as RunEntityWithChildren;

const SelectionHarness = ({ enableNewEvalRunsUi }: { enableNewEvalRunsUi: boolean }) => {
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  return (
    <>
      <ExperimentEvaluationRunsTable
        data={RUN_IDS.map(makeRun)}
        uniqueColumns={[]}
        selectedColumns={getEvalRunsBaseSelectionState(enableNewEvalRunsUi)}
        setSelectedRunUuid={jest.fn()}
        isLoading={false}
        hasNextPage={false}
        rowSelection={rowSelection}
        setRowSelection={setRowSelection}
        setSelectedDatasetWithRun={jest.fn()}
        setIsDrawerOpen={jest.fn()}
        viewMode={ExperimentEvaluationRunsPageMode.TRACES}
        enableNewEvalRunsUi={enableNewEvalRunsUi}
      />
      <div data-testid="selected-runs">{Object.keys(rowSelection).sort().join(',')}</div>
    </>
  );
};

const renderHarness = (enableNewEvalRunsUi: boolean) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ExperimentEvaluationRunsRowVisibilityProvider>
          <TestRouter routes={[testRoute(<SelectionHarness enableNewEvalRunsUi={enableNewEvalRunsUi} />)]} />
        </ExperimentEvaluationRunsRowVisibilityProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

const getRowCheckbox = (runUuid: string) => screen.getByTestId(`eval-runs-table-cell-checkbox-${runUuid}`);
const getSelectedRuns = () => screen.getByTestId('selected-runs').textContent;

const shiftClick = async (element: HTMLElement) => {
  const user = userEvent.setup();
  await user.keyboard('{Shift>}');
  await user.click(element);
  await user.keyboard('{/Shift}');
};

describe('eval runs row selection', () => {
  test('flag on: shift-click selects the range from the last clicked row', async () => {
    renderHarness(true);

    await userEvent.click(getRowCheckbox('run-1'));
    await shiftClick(getRowCheckbox('run-3'));

    expect(getSelectedRuns()).toBe('run-1,run-2,run-3');
  });

  test('flag on: shift-click on a selected row clears the range', async () => {
    renderHarness(true);

    await userEvent.click(getRowCheckbox('run-1'));
    await shiftClick(getRowCheckbox('run-4'));
    await shiftClick(getRowCheckbox('run-2'));

    expect(getSelectedRuns()).toBe('run-1');
  });

  test('flag on: shift-click without a selected anchor toggles only the clicked row and anchors the next range', async () => {
    renderHarness(true);

    await userEvent.click(getRowCheckbox('run-1'));
    await userEvent.click(getRowCheckbox('run-1'));
    await shiftClick(getRowCheckbox('run-4'));
    expect(getSelectedRuns()).toBe('run-4');

    // The clicked row becomes the anchor for the next range.
    await shiftClick(getRowCheckbox('run-2'));
    expect(getSelectedRuns()).toBe('run-2,run-3,run-4');
  });

  test('flag on: the header checkbox selects and clears every run', async () => {
    renderHarness(true);

    await userEvent.click(screen.getByRole('checkbox', { name: 'Select all runs' }));
    expect(getSelectedRuns()).toBe(RUN_IDS.join(','));

    await userEvent.click(screen.getByRole('checkbox', { name: 'Select all runs' }));
    expect(getSelectedRuns()).toBe('');
  });

  test('flag off: shift-click toggles only the clicked row and there is no select-all checkbox', async () => {
    renderHarness(false);

    await userEvent.click(getRowCheckbox('run-1'));
    await shiftClick(getRowCheckbox('run-3'));

    expect(getSelectedRuns()).toBe('run-1,run-3');
    expect(screen.queryByTestId('eval-runs-table-select-all-checkbox')).not.toBeInTheDocument();
  });
});
