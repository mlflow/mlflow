import { renderHook } from '@testing-library/react';
import { useGetExperimentPageActiveTabByRoute } from './useGetExperimentPageActiveTabByRoute';
import { ExperimentPageTabName } from '../../../constants';

// Mock the modules
jest.mock('../../../../common/utils/FeatureUtils', () => ({
  shouldEnableExperimentPageChildRoutes: jest.fn(),
}));

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  useLocation: jest.fn(),
  matchPath: jest.fn((routePath, pathname) => {
    // Simple implementation of matchPath for testing
    if (routePath.includes(':experimentId')) {
      const routePattern = routePath.replace(':experimentId', '\\d+');
      const regex = new RegExp(routePattern);
      return regex.test(pathname);
    }
    return routePath === pathname;
  }),
  createMLflowRoutePath: jest.fn((path) => path),
}));

// Import the mocked modules
import { shouldEnableExperimentPageChildRoutes } from '../../../../common/utils/FeatureUtils';
import { useLocation } from '../../../../common/utils/RoutingUtils';

describe('useGetExperimentPageActiveTabByRoute', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Enable the feature flag by default
    jest.mocked(shouldEnableExperimentPageChildRoutes).mockReturnValue(true);
  });

  const testCases = [
    {
      name: 'should return Runs tab when on runs route',
      pathname: '/experiments/123/runs',
      expectedTabName: ExperimentPageTabName.Runs,
      expectedTopLevelTabName: ExperimentPageTabName.Runs,
    },
    {
      name: 'should return Traces tab when on traces route',
      pathname: '/experiments/123/traces',
      expectedTabName: ExperimentPageTabName.Traces,
      expectedTopLevelTabName: ExperimentPageTabName.Traces,
    },
    {
      name: 'should return Models tab when on models route',
      pathname: '/experiments/123/models',
      expectedTabName: ExperimentPageTabName.Models,
      expectedTopLevelTabName: ExperimentPageTabName.Models,
    },
    {
      name: 'should return undefined when on unknown route',
      pathname: '/experiments/123/unknown',
      expectedTabName: undefined,
      expectedTopLevelTabName: undefined,
    },
    {
      name: 'should return undefined when on experiment root route',
      pathname: '/experiments/123',
      expectedTabName: undefined,
      expectedTopLevelTabName: undefined,
    },
  ];

  test.each(testCases)('$name', ({ pathname, expectedTabName, expectedTopLevelTabName }) => {
    jest.mocked(useLocation).mockReturnValue({ pathname, state: undefined, search: '', hash: '', key: '' });

    const { result } = renderHook(() => useGetExperimentPageActiveTabByRoute());

    expect(result.current.tabName).toBe(expectedTabName);
    expect(result.current.topLevelTabName).toBe(expectedTopLevelTabName);
  });

  test('should return undefined when feature flag is disabled', () => {
    jest.mocked(shouldEnableExperimentPageChildRoutes).mockReturnValue(false);
    jest
      .mocked(useLocation)
      .mockReturnValue({ pathname: '/experiments/123/runs', state: undefined, search: '', hash: '', key: '' });

    const { result } = renderHook(() => useGetExperimentPageActiveTabByRoute());

    expect(result.current.tabName).toBeUndefined();
    expect(result.current.topLevelTabName).toBeUndefined();
  });
});
