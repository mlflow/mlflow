import React from 'react';
import { shallow } from 'enzyme';
import { ParallelCoordinatesPlotControls } from './ParallelCoordinatesPlotControls';

describe('unit tests', () => {
  let wrapper;
  let minimumProps;

  beforeEach(() => {
    minimumProps = {
      paramKeys: ['param_0', 'param_1'],
      metricKeys: ['metric_0', 'metric_1'],
      selectedParamKeys: ['param_0', 'param_1'],
      selectedMetricKeys: ['metric_0', 'metric_1'],
      handleParamsSelectChange: jest.fn(),
      handleMetricsSelectChange: jest.fn(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ParallelCoordinatesPlotControls {...minimumProps} />);
    expect(wrapper.length).toBe(1);
  });
});
