import React from 'react';
import { shallow } from 'enzyme';
import qs from 'qs';
import Fixtures from '../utils/test-utils/Fixtures';
import { MetricViewImpl } from './MetricView';
import Utils from '../../common/utils/Utils';
import MetricsPlotPanel from './MetricsPlotPanel';
import { PageHeader } from '../../shared/building_blocks/PageHeader';

const createLocation = (experimentIds, runUuids, metricKey) => {
  return {
    search:
      '?' +
      qs.stringify({
        experiments: experimentIds,
        runs: JSON.stringify(runUuids),
        plot_metric_keys: JSON.stringify([metricKey]),
      }),
  };
};

describe('MetricView', () => {
  let wrapper;
  let minimalProps;
  let experiments;
  let experimentIds;

  beforeEach(() => {
    experimentIds = ['2'];
    experiments = experimentIds.map((experimentId) =>
      Fixtures.createExperiment({
        experiment_id: experimentId.toString(),
        name: experimentId.toString(),
        lifecycle_stage: 'active',
      }),
    );

    minimalProps = {
      experiments,
      experimentIds,
      runUuids: [],
      runNames: [],
      metricKey: 'metricKey',
      location: createLocation(experimentIds, [''], 'metricKey'),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<MetricViewImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(PageHeader).props().title).toContain('metricKey');
  });

  test('should render sub-components', () => {
    const props = {
      ...minimalProps,
      runUuids: ['a', 'b', 'c'],
      runNames: ['d', 'e', 'f'],
    };

    Utils.getMetricPlotStateFromUrl = jest.fn(() => {
      return { selectedMetricKeys: ['selectedKey'] };
    });

    wrapper = shallow(<MetricViewImpl {...props} />);

    const pageHeaderTitle = wrapper.find(PageHeader);
    const { title } = pageHeaderTitle.props();
    expect(title).toContain('selectedKey');

    const metricsPlotPanel = wrapper.find(MetricsPlotPanel);
    expect(metricsPlotPanel.length).toBe(1);
    expect(metricsPlotPanel.props().experimentIds).toEqual(['2']);
    expect(metricsPlotPanel.props().runUuids).toEqual(['a', 'b', 'c']);
    expect(metricsPlotPanel.props().metricKey).toBe('metricKey');
  });
});
