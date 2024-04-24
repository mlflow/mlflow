import { act, renderHook } from '@testing-library/react-for-react-18';
import { useInitializeUIState } from './useInitializeUIState';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { loadExperimentViewState } from '../utils/persistSearchFacets';
import {
  type ExperimentPageUIStateV2,
  createExperimentPageUIStateV2,
  RUNS_VISIBILITY_MODE,
} from '../models/ExperimentPageUIStateV2';
import { createExperimentPageSearchFacetsStateV2 } from '../models/ExperimentPageSearchFacetsStateV2';
import { shouldEnableShareExperimentViewByTags } from '../../../../common/utils/FeatureUtils';
import { RunsChartType } from '../../runs-charts/runs-charts.types';

const experimentIds = ['experiment_1'];

jest.mock('../utils/persistSearchFacets');
jest.mock('../../../../common/utils/FeatureUtils');

const initialUIState = createExperimentPageUIStateV2();

describe('useInitializeUIState', () => {
  beforeEach(() => {
    jest.mocked(loadExperimentViewState).mockImplementation(() => ({}));
    jest.mocked(shouldEnableShareExperimentViewByTags).mockImplementation(() => true);
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
    const persistedState = {
      ...createExperimentPageSearchFacetsStateV2(),
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
      selectedColumns: ['metrics.m2'],
    });
  });

  test('should properly update UI state using both setter patterns', () => {
    const { result } = renderParametrizedHook();
    const [, setUIState] = result.current;

    const customUIState: ExperimentPageUIStateV2 = {
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
    });
  });
});
