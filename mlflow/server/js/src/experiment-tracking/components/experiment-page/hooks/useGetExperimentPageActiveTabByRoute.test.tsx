import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useGetExperimentPageActiveTabByRoute } from './useGetExperimentPageActiveTabByRoute';
import { ExperimentPageTabName } from '../../../constants';
import { useLocation } from '../../../../common/utils/RoutingUtils';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  useLocation: jest.fn(),
  matchPath: jest.fn((routePath: string, pathname: string) => {
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

describe('useGetExperimentPageActiveTabByRoute', () => {
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
});
