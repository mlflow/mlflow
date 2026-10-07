import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useGetExperimentPageActiveTabByRoute } from './useGetExperimentPageActiveTabByRoute';
import { ExperimentPageTabName } from '../../../constants';
import { useLocation } from '../../../../common/utils/RoutingUtils';
import { shouldEnableSessionGrouping } from '@databricks/web-shared/genai-traces-table';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  useLocation: jest.fn(),
  matchPath: jest.fn((routePath: string, pathname: string) => {
    // Simple implementation of matchPath for testing
    if (routePath.includes(':')) {
      const routePattern = routePath.replace(/:[^/]+/g, '[^/]+');
      const regex = new RegExp(`^${routePattern}$`);
      return regex.test(pathname);
    }
    return routePath === pathname;
  }),
  createMLflowRoutePath: jest.fn((path) => path),
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  shouldEnableSessionGrouping: jest.fn(() => true),
}));

describe('useGetExperimentPageActiveTabByRoute', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableSessionGrouping).mockReturnValue(true);
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
      name: 'should return ReviewQueue tab when on review-queue route',
      pathname: '/experiments/123/review-queue',
      expectedTabName: ExperimentPageTabName.ReviewQueue,
      expectedTopLevelTabName: ExperimentPageTabName.ReviewQueue,
    },
    {
      name: 'should return Datasets tab when on a dataset detail route',
      pathname: '/experiments/123/datasets/dataset-456',
      expectedTabName: ExperimentPageTabName.Datasets,
      expectedTopLevelTabName: ExperimentPageTabName.Datasets,
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
    {
      name: 'should return Traces tab for a single chat session route when session grouping is enabled',
      pathname: '/experiments/123/chat-sessions/session-1',
      expectedTabName: ExperimentPageTabName.Traces,
      expectedTopLevelTabName: ExperimentPageTabName.Traces,
    },
  ];

  test.each(testCases)('$name', ({ pathname, expectedTabName, expectedTopLevelTabName }) => {
    jest.mocked(useLocation).mockReturnValue({ pathname, state: undefined, search: '', hash: '', key: '' });

    const { result } = renderHook(() => useGetExperimentPageActiveTabByRoute());

    expect(result.current.tabName).toBe(expectedTabName);
    expect(result.current.topLevelTabName).toBe(expectedTopLevelTabName);
  });

  test('should keep the SingleChatSession tab when session grouping is disabled', () => {
    jest.mocked(shouldEnableSessionGrouping).mockReturnValue(false);
    jest.mocked(useLocation).mockReturnValue({
      pathname: '/experiments/123/chat-sessions/session-1',
      state: undefined,
      search: '',
      hash: '',
      key: '',
    });

    const { result } = renderHook(() => useGetExperimentPageActiveTabByRoute());

    expect(result.current.tabName).toBe(ExperimentPageTabName.SingleChatSession);
  });
});
