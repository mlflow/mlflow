/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { ParallelCoordinatesPlotControls } from './ParallelCoordinatesPlotControls';

describe('unit tests', () => {
  let wrapper;
  let mininumProps: any;

  beforeEach(() => {
    mininumProps = {
      paramKeys: ['param_0', 'param_1'],
      metricKeys: ['metric_0', 'metric_1'],
      selectedParamKeys: ['param_0', 'param_1'],
      selectedMetricKeys: ['metric_0', 'metric_1'],
      handleParamsSelectChange: jest.fn(),
      handleMetricsSelectChange: jest.fn(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ParallelCoordinatesPlotControls {...mininumProps} />);
    expect(wrapper.length).toBe(1);
  });
});
