/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import {
  ParallelCoordinatesPlotView,
  generateAttributesForCategoricalDimension,
  createDimension,
  inferType,
  UNKNOWN_TERM,
} from './ParallelCoordinatesPlotView';

describe('unit tests', () => {
  let wrapper;
  let instance;
  let mininumProps: any;

  beforeEach(() => {
    mininumProps = {
      runUuids: ['runUuid_0', 'runUuid_1'],
      paramKeys: ['param_0', 'param_1'],
      metricKeys: ['metric_0', 'metric_1'],
      paramDimensions: [
        {
          label: 'param_0',
          values: [1, 2],
          tickformat: '.5f',
        },
        {
          label: 'param_1',
          values: [2, 3],
          tickformat: '.5f',
        },
      ],
      metricDimensions: [
        {
          label: 'metric_0',
          values: [1, 2],
          tickformat: '.5f',
        },
        {
          label: 'metric_1',
          values: [2, 3],
          tickformat: '.5f',
        },
      ],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ParallelCoordinatesPlotView {...mininumProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('getDerivedStateFromProps should return null when the selections do not change', () => {
    const props = {
      paramKeys: ['param_0', 'param_1'],
      metricKeys: ['metric_0', 'metric_1'],
    };
    // state with different order but same selections
    const state = {
      sequence: ['param_0', 'metric_0', 'metric_1', 'param_1'],
    };
    expect(ParallelCoordinatesPlotView.getDerivedStateFromProps(props, state)).toBe(null);
  });

  test('getDerivedStateFromProps should return state when the selections changes', () => {
    const props = {
      paramKeys: ['param_0', 'param_1'],
      metricKeys: ['metric_0', 'metric_1', 'metric_2'], // props comes with an extra metric_2
    };
    const state = {
      sequence: ['param_0', 'metric_0', 'metric_1', 'param_1'],
    };
    expect(ParallelCoordinatesPlotView.getDerivedStateFromProps(props, state)).toEqual({
      sequence: ['param_0', 'param_1', 'metric_0', 'metric_1', 'metric_2'],
    });
  });

  test('maybeUpdateStateForColorScale should trigger setState when last metric change', () => {
    wrapper = shallow(<ParallelCoordinatesPlotView {...mininumProps} />);
    instance = wrapper.instance();
    instance.findLastMetricFromState = jest.fn(() => 'metric_1');
    instance.setState = jest.fn();
    instance.maybeUpdateStateForColorScale(['metric_1', 'metric_0']); // rightmost metric changes
    expect(instance.setState).toHaveBeenCalled();
  });

  test('maybeUpdateStateForColorScale should not trigger setState when last metric stays', () => {
    wrapper = shallow(<ParallelCoordinatesPlotView {...mininumProps} />);
    instance = wrapper.instance();
    instance.findLastMetricFromState = jest.fn(() => 'metric_1');
    instance.setState = jest.fn();
    instance.maybeUpdateStateForColorScale(['metric_0', 'metric_1']); // rightmost metric stays
    expect(instance.setState).not.toHaveBeenCalled();
  });

  test('generateAttributesForCategoricalDimension', () => {
    expect(generateAttributesForCategoricalDimension(['A', 'B', 'C', 'B', 'C'])).toEqual({
      values: [0, 1, 2, 1, 2],
      tickvals: [0, 1, 2],
      ticktext: ['A', 'B', 'C'],
    });
  });

  test('inferType works with numeric dimension', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 1 },
      },
      runUuid_1: {
        metric_0: { value: 2 },
      },
    };
    expect(inferType(key, runUuids, entryByRunUuid)).toBe('number');
  });

  test('inferType works with categorical dimension', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 'B' },
      },
      runUuid_1: {
        metric_0: { value: 'A' },
      },
    };
    expect(inferType(key, runUuids, entryByRunUuid)).toBe('string');
  });

  test('inferType works with numeric dimension that includes NaNs', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: NaN },
      },
      runUuid_1: {
        metric_0: { value: NaN },
      },
    };
    expect(inferType(key, runUuids, entryByRunUuid)).toBe('number');
  });

  test('inferType works with numeric dimension specified as strings', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: '1.0' },
      },
      runUuid_1: {
        metric_0: { value: 'NaN' },
      },
    };
    expect(inferType(key, runUuids, entryByRunUuid)).toBe('number');
  });

  test('inferType works with mixed string and number dimension', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: '1.0' },
      },
      runUuid_1: {
        metric_0: { value: 'this thing is a string' },
      },
    };
    expect(inferType(key, runUuids, entryByRunUuid)).toBe('string');
  });

  test('createDimension should work with numeric dimension', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 1 },
      },
      runUuid_1: {
        metric_0: { value: 2 },
      },
    };
    expect(createDimension(key, runUuids, entryByRunUuid)).toEqual({
      label: 'metric_0',
      values: [1, 2],
      tickformat: '.5f',
    });
  });

  test('createDimension should work with categorical dimension', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 'B' },
      },
      runUuid_1: {
        metric_0: { value: 'A' },
      },
    };
    expect(createDimension(key, runUuids, entryByRunUuid)).toEqual({
      label: 'metric_0',
      values: [1, 0],
      tickvals: [0, 1],
      ticktext: ['A', 'B'],
    });
  });

  test('createDimension should work with missing values and fill in NaN', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 1 },
      },
      runUuid_1: {},
    };
    expect(createDimension(key, runUuids, entryByRunUuid)).toEqual({
      label: 'metric_0',
      values: [1, 1.01],
      tickformat: '.5f',
    });
  });

  test('createDimension should work with NaN and fill in NaN', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 1 },
      },
      runUuid_1: {
        metric_0: { value: NaN },
      },
    };
    expect(createDimension(key, runUuids, entryByRunUuid)).toEqual({
      label: 'metric_0',
      values: [1, NaN],
      tickformat: '.5f',
    });
  });

  test('createDimension should work with undefined values for strings series', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 'True' },
      },
      runUuid_1: {},
    };
    expect(createDimension(key, runUuids, entryByRunUuid)).toEqual({
      label: 'metric_0',
      ticktext: ['True', UNKNOWN_TERM],
      tickvals: [0, 1],
      values: [0, 1],
    });
  });

  test('createDimension should work with undefined values for number series', () => {
    const key = 'metric_0';
    const runUuids = ['runUuid_0', 'runUuid_1'];
    const entryByRunUuid = {
      runUuid_0: {
        metric_0: { value: 1 },
      },
      runUuid_1: {},
    };
    expect(createDimension(key, runUuids, entryByRunUuid)).toEqual({
      label: 'metric_0',
      values: [1, 1.01],
      tickformat: '.5f',
    });
  });

  test('getColorScaleConfigsForDimension', () => {
    wrapper = shallow(<ParallelCoordinatesPlotView {...mininumProps} />);
    instance = wrapper.instance();
    const dimension = {
      label: 'metric_0',
      values: [3, 1, 2, 3, 0, 2],
    };
    expect(ParallelCoordinatesPlotView.getColorScaleConfigsForDimension(dimension)).toEqual({
      showscale: true,
      colorscale: 'Jet',
      cmin: 0,
      cmax: 3,
      color: [3, 1, 2, 3, 0, 2],
    });
  });
});
