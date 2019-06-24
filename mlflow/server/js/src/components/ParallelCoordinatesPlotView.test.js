import React from 'react';
import { shallow } from 'enzyme';
import { ParallelCoordinatesPlotView } from './ParallelCoordinatesPlotView';

describe('unit tests', () => {
  let wrapper;
  let instance;
  let mininumProps;

  beforeEach(() => {
    mininumProps = {
      runUuids: ['runUuid_0', 'runUuid_1'],
      paramKeys: ['param_0', 'param_1'],
      metricKeys: ['metric_0', 'metric_1'],
      paramDimensions: [
        {
          label: 'param_0',
          values: [1, 2],
        },
        {
          label: 'param_1',
          values: [2, 3],
        },
      ],
      metricDimensions: [
        {
          label: 'metric_0',
          values: [1, 2],
        },
        {
          label: 'metric_1',
          values: [2, 3],
        },
      ],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ParallelCoordinatesPlotView {...mininumProps}/>);
    expect(wrapper.length).toBe(1);
  });
});
