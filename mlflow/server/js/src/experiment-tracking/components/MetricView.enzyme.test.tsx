/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import qs from 'qs';
import Fixtures from '../utils/test-utils/Fixtures';
import { MetricViewImpl } from './MetricView';
import Utils from '../../common/utils/Utils';
import MetricsPlotPanel from './MetricsPlotPanel';
import { PageHeader } from '../../shared/building_blocks/PageHeader';

const createLocation = (experimentIds: any, runUuids: any, metricKey: any) => {
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
  let minimalProps: any;
  let experiments;
  let experimentIds;

  beforeEach(() => {
    experimentIds = ['2'];
    experiments = experimentIds.map((experimentId) =>
      Fixtures.createExperiment({
        experimentId: experimentId.toString(),
        name: experimentId.toString(),
        lifecycleStage: 'active',
      }),
    );

    minimalProps = {
      experiments,
      experimentIds,
      runUuids: [],
      runNames: [],
      metricKey: 'metricKey',
      location: createLocation(experimentIds, [''], 'metricKey'),
      navigate: jest.fn(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<MetricViewImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(PageHeader).props().title).toContain('metricKey');
  });

  test('should render sub-components', () => {
    const mockNavigate = jest.fn();
    const props = {
      ...minimalProps,
      runUuids: ['a', 'b', 'c'],
      runNames: ['d', 'e', 'f'],
      navigate: mockNavigate,
    };

    // @ts-expect-error TS(2322): Type 'Mock<{ selectedMetricKeys: string[]; }, []>'... Remove this comment to see the full error message
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
    expect(metricsPlotPanel.props().location).toEqual(props.location);
    expect(metricsPlotPanel.props().navigate).toBe(mockNavigate);
  });
});
