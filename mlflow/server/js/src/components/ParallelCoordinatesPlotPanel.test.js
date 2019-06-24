import React from 'react';
import { shallow } from 'enzyme';
import { ParallelCoordinatesPlotPanel } from './ParallelCoordinatesPlotPanel';

describe('unit tests', () => {
  let wrapper;
  let mininumProps;

  beforeEach(() => {
    mininumProps = {
      runUuids: ['runUuid_0', 'runUuid_1'],
      sharedParamKeys: ['param_0', 'param_1'],
      sharedMetricKeys: ['metric_0', 'metric_1'],
    }
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ParallelCoordinatesPlotPanel {...mininumProps}/>);
    expect(wrapper.length).toBe(1);
  });
});
