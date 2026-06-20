import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { generateExperimentHash, useInitializeUIState } from './useInitializeUIState';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { loadExperimentViewState } from '../utils/persistSearchFacets';
import {
  type ExperimentPageUIState,
  createExperimentPageUIState,
  RUNS_VISIBILITY_MODE,
  migrateSelectedColumns,
} from '../models/ExperimentPageUIState';
import { ATTRIBUTE_COLUMN_LABELS, COLUMN_TYPES } from '../../../constants';
import { makeCanonicalSortKey } from '../utils/experimentPage.common-utils';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { RunsChartType } from '../../runs-charts/runs-charts.types';
import { expandedEvaluationRunRowsUIStateInitializer } from '../utils/expandedRunsViewStateInitializer';
import { createBaseExperimentEntity, createBaseRunsData, createBaseRunsInfoEntity } from '../utils/test-utils';

const experimentIds = ['experiment_1'];

jest.mock('../utils/persistSearchFacets');
jest.mock('../utils/expandedRunsViewStateInitializer', () => ({
  expandedEvaluationRunRowsUIStateInitializer: jest.fn(),
}));

const initialUIState = createExperimentPageUIState();
const baseRunsData = createBaseRunsData();
const experiment1 = createBaseExperimentEntity();
const runInfoEntity1 = createBaseRunsInfoEntity();

describe('useInitializeUIState', () => {
  beforeEach(() => {
    jest.mocked(loadExperimentViewState).mockImplementation(() => ({}));
  });

  const renderParametrizedHook = () => {
    return renderHook(() => useInitializeUIState(experimentIds), {
      wrapper: ({ children }) => <MemoryRouter>{children}</MemoryRouter>,
    });
  };

  test('should return empty UI state when no persisted state is present', () => {
    const { result } = renderParametrizedHook();
    const [uiState] = result.current;
    expect(uiState).toEqual(initialUIState);
  });

  test('should return persisted UI state when present', () => {
    const DATE_KEY = makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DATE);
    const DURATION_KEY = makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DURATION);
    const persistedState = {
      ...createExperimentPageSearchFacetsState(),
      ...initialUIState,
      orderByKey: 'metrics.m1',
      orderByAsc: true,
      viewMaximized: true,
      runListHidden: true,
      selectedColumns: ['metrics.m2'],
    };
    jest.mocked(loadExperimentViewState).mockImplementation(() => persistedState);
    const { result } = renderParametrizedHook();
    const [uiState] = result.current;
    expect(uiState).toEqual({
      ...initialUIState,
      viewMaximized: true,
      runListHidden: true,
      // Migration adds Date and Duration columns if missing
      selectedColumns: [DATE_KEY, DURATION_KEY, 'metrics.m2'],
    });
  });

  test('should properly update UI state using both setter patterns', () => {
    const { result } = renderParametrizedHook();
    const [, setUIState] = result.current;

    const customUIState: ExperimentPageUIState = {
      runListHidden: true,
      runsPinned: ['run_1'],
      selectedColumns: ['metrics.m2'],
      viewMaximized: true,
      runsExpanded: { run_2: true },
      runsHidden: ['run_3'],
      runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
      compareRunCharts: [{ type: RunsChartType.BAR, deleted: false, isGenerated: true }],
      isAccordionReordered: false,
      groupBy: '',
      groupsExpanded: {},
      autoRefreshEnabled: true,
    };

    act(() => {
      setUIState(customUIState);
    });

    expect(result.current[0]).toEqual(customUIState);

    act(() => {
      setUIState((prevState) => ({
        ...prevState,
        viewMaximized: false,
        runsExpanded: { run_4: true },
        compareRunCharts: [],
      }));
    });

    expect(result.current[0]).toEqual({
      runListHidden: true,
      runsPinned: ['run_1'],
      selectedColumns: ['metrics.m2'],
      viewMaximized: false,
      runsExpanded: { run_4: true },
      runsHidden: ['run_3'],
      runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
      compareRunCharts: [],
      isAccordionReordered: false,
      groupBy: '',
      groupsExpanded: {},
      autoRefreshEnabled: true,
    });
  });

  describe('seedInitialUIState', () => {
    beforeEach(() => {
      jest.clearAllMocks();
    });

    const runsData = { ...baseRunsData, runInfos: [runInfoEntity1] };

    test('should not seed UI state if there are no experiments or runs', () => {
      const { result } = renderParametrizedHook();
      const [, , seedInitialUIState] = result.current;

      act(() => {
        seedInitialUIState([], runsData);
      });

      expect(result.current[0]).toEqual(initialUIState);
      expect(expandedEvaluationRunRowsUIStateInitializer).not.toHaveBeenCalled();
    });

    test("should not trigger uiStateInitializers if it's not the first session", () => {
      const persistedState = {
        ...createExperimentPageSearchFacetsState(),
        ...initialUIState,
        orderByKey: 'metrics.m1',
        orderByAsc: true,
        viewMaximized: true,
        runListHidden: true,
        selectedColumns: ['metrics.m2'],
      };
      jest.mocked(loadExperimentViewState).mockImplementation(() => persistedState);

      const { result } = renderParametrizedHook();
      const [, , seedInitialUIState] = result.current;

      act(() => {
        seedInitialUIState([experiment1], runsData);
      });

      expect(expandedEvaluationRunRowsUIStateInitializer).not.toHaveBeenCalled();
    });

    test('should not trigger uiStateInitializers if there are no new jobs', async () => {
      const { result } = renderParametrizedHook();

      act(() => {
        result.current[2]([experiment1], runsData);
      });

      act(() => {
        result.current[2]([experiment1], runsData);
      });

      expect(expandedEvaluationRunRowsUIStateInitializer).toHaveBeenCalledTimes(1);
    });

    test('should trigger uiStateInitializers if there are new runs', async () => {
      const { result } = renderParametrizedHook();

      act(() => {
        result.current[2]([experiment1], runsData);
      });

      act(() => {
        result.current[2]([experiment1], {
          ...runsData,
          runInfos: [...runsData.runInfos, { ...runInfoEntity1, runUuid: 'run_2' }],
        });
      });

      expect(expandedEvaluationRunRowsUIStateInitializer).toHaveBeenCalledTimes(2);
    });

    test('should not trigger uiStateInitializers if non-unique run ids are sorted differently', async () => {
      const { result } = renderParametrizedHook();

      act(() => {
        result.current[2]([experiment1], {
          ...runsData,
          runInfos: [...runsData.runInfos, { ...runInfoEntity1, runUuid: 'run_2' }],
        });
      });

      act(() => {
        result.current[2]([experiment1], {
          ...runsData,
          runInfos: [{ ...runInfoEntity1, runUuid: 'run_2' }, ...runsData.runInfos],
        });
      });

      expect(expandedEvaluationRunRowsUIStateInitializer).toHaveBeenCalledTimes(1);
    });

    test('should trigger uiStateInitializers', () => {
      const { result } = renderParametrizedHook();
      const [, , seedInitialUIState] = result.current;

      act(() => {
        seedInitialUIState([experiment1], runsData);
      });

      const initializerInput = [[experiment1], initialUIState, runsData, false];

      // @ts-expect-error A spread argument must either have a tuple type or be passed to a rest parameter
      expect(expandedEvaluationRunRowsUIStateInitializer).toHaveBeenCalledWith(...initializerInput);
    });

    test('should trigger uiStateInitializers with isSeeded = true on 2nd invocation', () => {
      const { result } = renderParametrizedHook();
      jest.mocked(expandedEvaluationRunRowsUIStateInitializer).mockReturnValue(initialUIState);

      act(() => {
        result.current[2]([experiment1], runsData);
      });

      expect(expandedEvaluationRunRowsUIStateInitializer).toHaveBeenCalledWith(
        [experiment1],
        initialUIState,
        runsData,
        false,
      );

      act(() => {
        result.current[2]([experiment1], {
          ...runsData,
          runInfos: [...runsData.runInfos, { ...runInfoEntity1, runUuid: 'run_2' }],
        });
      });

      expect(expandedEvaluationRunRowsUIStateInitializer).toHaveBeenCalledWith(
        [experiment1],
        initialUIState,
        expect.objectContaining({ runInfos: expect.any(Array) }),
        true,
      );
    });
  });
});

describe('generateExperimentHash', () => {
  test('it generates a hash key based on the experiment and run data', () => {
    const runs = {
      ...baseRunsData,
      runInfos: [runInfoEntity1, { ...runInfoEntity1, runUuid: 'run_2' }],
    };
    expect(generateExperimentHash(runs, [experiment1])).toEqual('experiment_1:run_1:run_2');
  });

  test("returns null if there's no runs", () => {
    expect(generateExperimentHash(baseRunsData, [experiment1])).toBeNull();
  });

  test("returns null if there's no experiments", () => {
    const runs = {
      ...baseRunsData,
      runInfos: [runInfoEntity1],
    };
    expect(generateExperimentHash(runs, [])).toBeNull();
  });

  test('sorts experiments and runs before generating hash', () => {
    const runs = {
      ...baseRunsData,
      runInfos: [
        { ...runInfoEntity1, runUuid: 'run_3' },
        { ...runInfoEntity1, runUuid: 'run_1' },
        { ...runInfoEntity1, runUuid: 'run_2' },
      ],
    };
    const experiments = [
      { ...experiment1, experimentId: 'experiment_2' },
      { ...experiment1, experimentId: 'experiment_1' },
    ];

    expect(generateExperimentHash(runs, experiments)).toEqual('experiment_1:experiment_2:run_1:run_2:run_3');
  });
});

describe('migrateSelectedColumns', () => {
  const DATE_COLUMN_KEY = makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DATE);
  const DURATION_COLUMN_KEY = makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DURATION);

  test('returns unchanged array if Date and Duration columns already exist', () => {
    const selectedColumns = [DATE_COLUMN_KEY, DURATION_COLUMN_KEY, 'metrics.m1'];
    const result = migrateSelectedColumns(selectedColumns);
    expect(result).toEqual(selectedColumns);
  });

  test('adds Date column if missing', () => {
    const selectedColumns = [DURATION_COLUMN_KEY, 'metrics.m1'];
    const result = migrateSelectedColumns(selectedColumns);
    expect(result).toEqual([DATE_COLUMN_KEY, DURATION_COLUMN_KEY, 'metrics.m1']);
  });

  test('adds Duration column if missing', () => {
    const selectedColumns = [DATE_COLUMN_KEY, 'metrics.m1'];
    const result = migrateSelectedColumns(selectedColumns);
    // Duration is added at the beginning with unshift
    expect(result).toEqual([DURATION_COLUMN_KEY, DATE_COLUMN_KEY, 'metrics.m1']);
  });

  test('adds both Date and Duration columns if both are missing', () => {
    const selectedColumns = ['metrics.m1', 'params.p1'];
    const result = migrateSelectedColumns(selectedColumns);
    expect(result).toEqual([DATE_COLUMN_KEY, DURATION_COLUMN_KEY, 'metrics.m1', 'params.p1']);
  });

  test('adds columns at the beginning of the array', () => {
    const selectedColumns = ['attributes.`Source`', 'attributes.`Models`'];
    const result = migrateSelectedColumns(selectedColumns);
    expect(result[0]).toBe(DATE_COLUMN_KEY);
    expect(result[1]).toBe(DURATION_COLUMN_KEY);
  });
});

describe('useInitializeUIState migration', () => {
  const DATE_COLUMN_KEY = makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DATE);
  const DURATION_COLUMN_KEY = makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DURATION);

  const renderParametrizedHook = () => {
    return renderHook(() => useInitializeUIState(experimentIds), {
      wrapper: ({ children }) => <MemoryRouter>{children}</MemoryRouter>,
    });
  };

  test('should migrate persisted state to include Date and Duration columns', () => {
    // Simulate old persisted state without Date and Duration columns
    const oldPersistedState = {
      ...initialUIState,
      selectedColumns: ['attributes.`Source`', 'attributes.`Models`', 'metrics.m1'],
    };
    jest.mocked(loadExperimentViewState).mockImplementation(() => oldPersistedState);

    const { result } = renderParametrizedHook();
    const [uiState] = result.current;

    // Verify Date and Duration were added
    expect(uiState.selectedColumns).toContain(DATE_COLUMN_KEY);
    expect(uiState.selectedColumns).toContain(DURATION_COLUMN_KEY);
    // Verify original columns are preserved
    expect(uiState.selectedColumns).toContain('attributes.`Source`');
    expect(uiState.selectedColumns).toContain('attributes.`Models`');
    expect(uiState.selectedColumns).toContain('metrics.m1');
  });
});
