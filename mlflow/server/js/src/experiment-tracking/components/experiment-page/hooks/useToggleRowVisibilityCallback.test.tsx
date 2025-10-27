import { renderHook, act } from '@testing-library/react';
import { useToggleRowVisibilityCallback } from './useToggleRowVisibilityCallback';
import { ExperimentPageUIStateContextProvider } from '../contexts/ExperimentPageUIStateContext';
import { RUNS_VISIBILITY_MODE, createExperimentPageUIState } from '../models/ExperimentPageUIState';
import { useEffect, useState } from 'react';
import type { RunRowType } from '../utils/experimentPage.row-types';
import {
  shouldEnableToggleIndividualRunsInGroups,
  shouldUseRunRowsVisibilityMap,
} from '../../../../common/utils/FeatureUtils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldEnableToggleIndividualRunsInGroups: jest.fn(),
  shouldUseRunRowsVisibilityMap: jest.fn(() => false),
}));

describe('useToggleRowVisibilityCallback', () => {
  let currentUIState = createExperimentPageUIState();
  const renderConfiguredHook = (
    tableRows: RunRowType[] = [],
    initialUiState = createExperimentPageUIState(),
    useGroupedValuesInCharts = true,
  ) =>
    renderHook((props) => useToggleRowVisibilityCallback(props.tableRows, useGroupedValuesInCharts), {
      initialProps: { tableRows },
      wrapper: function Wrapper({ children }) {
        const [uiState, setUIState] = useState(initialUiState);
        useEffect(() => {
          currentUIState = uiState;
        }, [uiState]);
        return (
          <ExperimentPageUIStateContextProvider setUIState={setUIState}>
            {children}
          </ExperimentPageUIStateContextProvider>
        );
      },
    }).result.current;
  test('performs simple update of the mode in the UI state', () => {
    const toggleRowVisibility = renderConfiguredHook();

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.SHOWALL);
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.SHOWALL);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.HIDEALL);
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.HIDEALL);
  });

  test('enables certain run row in the UI state using legacy runsHidden UI state', () => {
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(false);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: true, runUuid: 'run-1' },
        { hidden: false, runUuid: 'run-2' },
        { hidden: true, runUuid: 'run-3' },
        { hidden: false, runUuid: 'run-4' },
        { hidden: false, runUuid: 'run-5' },
      ] as any,
      { ...createExperimentPageUIState(), runsHidden: ['run-1', 'run-3'] },
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'run-5');
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.CUSTOM);
    // Assert updated hidden runs
    expect(currentUIState.runsHidden).toEqual(['run-1', 'run-3', 'run-5']);
  });

  test('enables certain run row in the UI state using runsVisibilityMap UI state', () => {
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(true);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: true, runUuid: 'run-1' },
        { hidden: false, runUuid: 'run-2' },
        { hidden: true, runUuid: 'run-3' },
        { hidden: false, runUuid: 'run-4' },
        { hidden: false, runUuid: 'run-5' },
      ] as any,
      { ...createExperimentPageUIState(), runsVisibilityMap: { 'run-1': false, 'run-3': false } },
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'run-5', false);
    });

    // Assert mode didn't change
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);
    // Assert updated runs visibility map
    expect(currentUIState.runsVisibilityMap).toEqual({ 'run-1': false, 'run-3': false, 'run-5': true });
  });

  test('disables run group when useGroupedValuesInCharts is true (using legacy runsHidden UI state)', () => {
    jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockReturnValue(true);
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(false);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: false, rowUuid: 'group-1', groupParentInfo: { runUuids: ['run-1-a', 'run-1-b'] } },
        { hidden: false, runUuid: 'run-1-a' },
        { hidden: false, runUuid: 'run-1-b' },
        { hidden: false, rowUuid: 'group-2', groupParentInfo: { runUuids: ['run-2-a', 'run-2-b'] } },
        { hidden: false, runUuid: 'run-2-a' },
        { hidden: false, runUuid: 'run-2-b' },
      ] as any,
      { ...createExperimentPageUIState() },
      true,
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'group-2');
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.CUSTOM);
    // Assert updated hidden runs
    expect(currentUIState.runsHidden).toEqual(['group-2']);
  });

  test('disables run group when useGroupedValuesInCharts is true (using runsVisibilityMap UI state)', () => {
    jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockReturnValue(true);
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(true);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: false, rowUuid: 'group-1', groupParentInfo: { runUuids: ['run-1-a', 'run-1-b'] } },
        { hidden: false, runUuid: 'run-1-a' },
        { hidden: false, runUuid: 'run-1-b' },
        { hidden: false, rowUuid: 'group-2', groupParentInfo: { runUuids: ['run-2-a', 'run-2-b'] } },
        { hidden: false, runUuid: 'run-2-a' },
        { hidden: false, runUuid: 'run-2-b' },
      ] as any,
      { ...createExperimentPageUIState() },
      true,
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'group-2', false);
    });

    // Assert mode didn't change
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);
    // Assert updated runs visibility map
    expect(currentUIState.runsVisibilityMap).toEqual({ 'group-2': true });
  });

  test('disables run group when useGroupedValuesInCharts is false (using legacy runsHidden UI state)', () => {
    jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockReturnValue(true);
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(false);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: false, rowUuid: 'group-1', groupParentInfo: { runUuids: ['run-1-a', 'run-1-b'] } },
        { hidden: false, runUuid: 'run-1-a' },
        { hidden: false, runUuid: 'run-1-b' },
        { hidden: false, rowUuid: 'group-2', groupParentInfo: { runUuids: ['run-2-a', 'run-2-b'] } },
        { hidden: false, runUuid: 'run-2-a' },
        { hidden: false, runUuid: 'run-2-b' },
      ] as any,
      { ...createExperimentPageUIState() },
      false,
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'group-2');
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.CUSTOM);
    // Assert updated hidden runs
    expect(currentUIState.runsHidden).toEqual(['run-2-a', 'run-2-b']);
  });

  test('disables run group when useGroupedValuesInCharts is false (using runsVisibilityMap UI state)', () => {
    jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockReturnValue(true);
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(true);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: false, rowUuid: 'group-1', groupParentInfo: { runUuids: ['run-1-a', 'run-1-b'] } },
        { hidden: false, runUuid: 'run-1-a' },
        { hidden: false, runUuid: 'run-1-b' },
        { hidden: false, rowUuid: 'group-2', groupParentInfo: { runUuids: ['run-2-a', 'run-2-b'] } },
        { hidden: false, runUuid: 'run-2-a' },
        { hidden: false, runUuid: 'run-2-b' },
      ] as any,
      { ...createExperimentPageUIState() },
      false,
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'group-2');
    });

    // Assert mode didn't change
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);
    // Assert updated runs visibility map
    expect(currentUIState.runsVisibilityMap).toEqual({ 'run-2-a': true, 'run-2-b': true });
  });

  test('disables certain run row in the UI state when using legacy runsHidden UI state', () => {
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(false);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: true, runUuid: 'run-1' },
        { hidden: false, runUuid: 'run-2' },
        { hidden: true, runUuid: 'run-3' },
        { hidden: false, runUuid: 'run-4' },
        { hidden: false, runUuid: 'run-5' },
      ] as any,
      { ...createExperimentPageUIState(), runsHidden: ['run-1', 'run-3'] },
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'run-3');
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.CUSTOM);
    // Assert updated hidden runs
    expect(currentUIState.runsHidden).toEqual(['run-1']);
  });

  test('disables certain run row in the UI state when using runsVisibilityMap UI state', () => {
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(true);
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: true, runUuid: 'run-1' },
        { hidden: false, runUuid: 'run-2' },
        { hidden: true, runUuid: 'run-3' },
        { hidden: false, runUuid: 'run-4' },
        { hidden: false, runUuid: 'run-5' },
      ] as any,
      {
        ...createExperimentPageUIState(),
        runsVisibilityMap: { 'run-1': false, 'run-3': true },
      },
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'run-3', true);
    });

    // Assert mode didn't change
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIState().runsHiddenMode);
    // Assert updated runs visibility map
    expect(currentUIState.runsVisibilityMap).toEqual({ 'run-1': false, 'run-3': false });
  });
});
