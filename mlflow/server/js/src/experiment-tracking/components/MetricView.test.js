import React from 'react';
import { shallow } from 'enzyme';
import qs from 'qs';
import Fixtures from '../utils/test-utils/Fixtures';
import { MetricViewImpl } from './MetricView';
import { BreadcrumbTitle } from './BreadcrumbTitle';
import Utils from '../../common/utils/Utils';
import MetricsPlotPanel from './MetricsPlotPanel';

describe('MetricView', () => {
  let wrapper;
  let minimalProps;
  let experiment;

  const createLocation = (experimentId, runUuids, metricKey) => {
    return {
      search:
        '?' +
        qs.stringify({
          experiment: experimentId,
          runs: JSON.stringify(runUuids),
          plot_metric_keys: JSON.stringify([metricKey]),
        }),
    };
  };

  beforeEach(() => {
    experiment = Fixtures.createExperiment({
      experiment_id: '2',
      name: '2',
      lifecycle_stage: 'active',
    });
    minimalProps = {
      experiment,
      runUuids: [],
      runNames: [],
      metricKey: 'metricKey',
      location: createLocation(experiment.experiment_id, [''], 'metricKey'),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<MetricViewImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    expect(
      wrapper
        .find(BreadcrumbTitle)
        .dive()
        .text(),
    ).toContain('metricKey');
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

    const breadcrumbTitle = wrapper.find(BreadcrumbTitle);
    const titleSpan = shallow(breadcrumbTitle.props().title);
    expect(titleSpan.text()).toContain('selectedKey');

    const metricsPlotPanel = wrapper.find(MetricsPlotPanel);
    expect(metricsPlotPanel.length).toBe(1);
    expect(metricsPlotPanel.props().experimentId).toBe('2');
    expect(metricsPlotPanel.props().runUuids).toEqual(['a', 'b', 'c']);
    expect(metricsPlotPanel.props().metricKey).toBe('metricKey');
  });
});
