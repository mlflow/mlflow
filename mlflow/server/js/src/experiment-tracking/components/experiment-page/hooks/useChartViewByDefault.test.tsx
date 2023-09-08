import { UpdateExperimentSearchFacetsFn } from '../../../types';
import { useChartViewByDefault } from './useChartViewByDefault';
import { render } from '@testing-library/react';

describe('useChartViewByDefault', () => {
  const updateSearchFacetsMock = jest.fn();
  const TestComponent = ({
    isLoadingRuns = false,
    metricKeyList = [],
    updateSearchFacets = updateSearchFacetsMock,
  }: {
    isLoadingRuns?: boolean;
    metricKeyList?: string[];
    updateSearchFacets?: UpdateExperimentSearchFacetsFn;
  }) => {
    useChartViewByDefault(isLoadingRuns, metricKeyList, updateSearchFacets);
    return null;
  };

  beforeEach(() => {
    updateSearchFacetsMock.mockClear();
  });

  test('should not call update if loading runs', () => {
    render(<TestComponent isLoadingRuns />);

    expect(updateSearchFacetsMock).toBeCalledTimes(0);
  });

  test('should not call update if metrics key list is empty', () => {
    render(<TestComponent isLoadingRuns={false} />);

    expect(updateSearchFacetsMock).toBeCalledTimes(0);
  });

  test('should call update only once', () => {
    const component = render(
      <TestComponent isLoadingRuns={false} metricKeyList={['metric_key_1']} />,
    );
    const [updatedState] = updateSearchFacetsMock.mock.lastCall;

    expect(updateSearchFacetsMock).toBeCalledTimes(1);
    expect(updatedState).toEqual({ compareRunsMode: 'CHART' });

    updateSearchFacetsMock.mockClear();
    component.rerender(
      <TestComponent isLoadingRuns={false} metricKeyList={['metric_key_1', 'metric_key_2']} />,
    );

    expect(updateSearchFacetsMock).toBeCalledTimes(0);
  });
});
