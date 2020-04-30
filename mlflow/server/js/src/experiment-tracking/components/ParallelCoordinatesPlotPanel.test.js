import React from 'react';
import { shallow } from 'enzyme';
import { ParallelCoordinatesPlotPanel } from './ParallelCoordinatesPlotPanel';
import ParallelCoordinatesPlotView from './ParallelCoordinatesPlotView';
import { Empty } from 'antd';

describe('unit tests', () => {
  let wrapper;
  let instance;
  let mininumProps;

  beforeEach(() => {
    mininumProps = {
      runUuids: ['runUuid_0', 'runUuid_1'],
      sharedParamKeys: ['param_0', 'param_1'],
      sharedMetricKeys: ['metric_0', 'metric_1'],
      allParamKeys: ['param_0', 'param_1', 'param_2'],
      allMetricKeys: ['metric_0', 'metric_1', 'metric_2'],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ParallelCoordinatesPlotPanel {...mininumProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render empty component when no dimension is selected', () => {
    wrapper = shallow(<ParallelCoordinatesPlotPanel {...mininumProps} />);
    instance = wrapper.instance();
    expect(wrapper.find(ParallelCoordinatesPlotView)).toHaveLength(1);
    expect(wrapper.find(Empty)).toHaveLength(0);
    instance.setState({
      selectedParamKeys: [],
      selectedMetricKeys: [],
    });
    expect(wrapper.find(ParallelCoordinatesPlotView)).toHaveLength(0);
    expect(wrapper.find(Empty)).toHaveLength(1);
  });
});
