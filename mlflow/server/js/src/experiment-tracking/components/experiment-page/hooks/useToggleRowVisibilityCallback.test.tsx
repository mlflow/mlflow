import { renderHook, act } from '@testing-library/react-for-react-18';
import { useToggleRowVisibilityCallback } from './useToggleRowVisibilityCallback';
import { ExperimentPageUIStateContextProvider } from '../contexts/ExperimentPageUIStateContext';
import { RUNS_VISIBILITY_MODE, createExperimentPageUIStateV2 } from '../models/ExperimentPageUIStateV2';
import { useEffect, useState } from 'react';
import { RunRowType } from '../utils/experimentPage.row-types';

describe('useToggleRowVisibilityCallback', () => {
  let currentUIState = createExperimentPageUIStateV2();
  const renderConfiguredHook = (tableRows: RunRowType[] = [], initialUiState = createExperimentPageUIStateV2()) =>
    renderHook((props) => useToggleRowVisibilityCallback(props.tableRows), {
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
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIStateV2().runsHiddenMode);

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

  test('enables certain run row in the UI state', () => {
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: true, runUuid: 'run-1' },
        { hidden: false, runUuid: 'run-2' },
        { hidden: true, runUuid: 'run-3' },
        { hidden: false, runUuid: 'run-4' },
        { hidden: false, runUuid: 'run-5' },
      ] as any,
      { ...createExperimentPageUIStateV2(), runsHidden: ['run-1', 'run-3'] },
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIStateV2().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'run-5');
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.CUSTOM);
    // Assert updated hidden runs
    expect(currentUIState.runsHidden).toEqual(['run-1', 'run-3', 'run-5']);
  });

  test('disables certain run row in the UI state', () => {
    const toggleRowVisibility = renderConfiguredHook(
      [
        { hidden: true, runUuid: 'run-1' },
        { hidden: false, runUuid: 'run-2' },
        { hidden: true, runUuid: 'run-3' },
        { hidden: false, runUuid: 'run-4' },
        { hidden: false, runUuid: 'run-5' },
      ] as any,
      { ...createExperimentPageUIStateV2(), runsHidden: ['run-1', 'run-3'] },
    );

    // Assert initial mode
    expect(currentUIState.runsHiddenMode).toBe(createExperimentPageUIStateV2().runsHiddenMode);

    act(() => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, 'run-3');
    });

    // Assert updated mode
    expect(currentUIState.runsHiddenMode).toBe(RUNS_VISIBILITY_MODE.CUSTOM);
    // Assert updated hidden runs
    expect(currentUIState.runsHidden).toEqual(['run-1']);
  });
});
