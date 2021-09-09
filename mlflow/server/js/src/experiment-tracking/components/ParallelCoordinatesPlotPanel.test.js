import React from 'react';
import { shallow } from 'enzyme';
import { ParallelCoordinatesPlotPanel, getDiffParams } from './ParallelCoordinatesPlotPanel';
import ParallelCoordinatesPlotView from './ParallelCoordinatesPlotView';
import { Empty } from 'antd';

describe('unit tests', () => {
  let wrapper;
  let instance;
  let mininumProps;

  beforeEach(() => {
    mininumProps = {
      runUuids: ['runUuid_0', 'runUuid_1'],
      diffParamKeys: ['param_0', 'param_1'],
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

  test('should select differing params correctly', () => {
    const runUuids = ['runId1', 'runId2', 'runId3'];
    const allParamKeys = ['param1', 'param2', 'param3'];
    const paramsByRunId = {
      runId1: { param1: { value: 1 }, param2: { value: 2 }, param3: { value: 3 } },
      runId2: { param1: { value: 1 }, param2: { value: 4 } },
      runId3: { param1: { value: 1 }, param2: { value: 3 } },
    };
    expect(getDiffParams(allParamKeys, runUuids, paramsByRunId)).toEqual(['param2', 'param3']);
  });

  test('should select differing params correctly when only one param', () => {
    const runUuids = ['runId1', 'runId2', 'runId3'];
    const allParamKeys = ['param1'];
    const paramsByRunId = {
      runId1: { param1: { value: 1 } },
      runId2: {},
      runId3: {},
    };
    expect(getDiffParams(allParamKeys, runUuids, paramsByRunId)).toEqual(['param1']);
  });
});
