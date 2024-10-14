/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { CompareRunContour } from './CompareRunContour';
import { shallowWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('unit tests', () => {
  let wrapper: any;
  let instance;
  const runUuids = ['run_uuid_0', 'run_uuid_1', 'run_uuid_2'];
  const commonProps = {
    runUuids,
    runInfos: runUuids.map((runUuid) => ({
      runUuid,
      experimentId: '1',
    })),
    runDisplayNames: runUuids,
  };
  beforeEach(() => {});
  test('should render with minimal props without exploding', () => {
    const props = {
      ...commonProps,
      paramLists: [
        [
          { key: 'p1', value: 1 },
          { key: 'p2', value: 2 },
        ],
        [
          { key: 'p1', value: 3 },
          { key: 'p2', value: 4 },
        ],
        [
          { key: 'p1', value: 5 },
          { key: 'p2', value: 6 },
        ],
      ],
      metricLists: [[{ key: 'm1', value: 7 }], [{ key: 'm1', value: 8 }], [{ key: 'm1', value: 9 }]],
    };
    wrapper = shallow(<CompareRunContour {...props} />);
    expect(wrapper.length).toBe(1);
    instance = wrapper.instance();
    expect(instance.paramKeys).toEqual(['p1', 'p2']);
    expect(instance.metricKeys).toEqual(['m1']);
    expect(instance.state).toEqual({
      disabled: false,
      reverseColor: false,
      xaxis: { key: 'p1', isMetric: false },
      yaxis: { key: 'p2', isMetric: false },
      zaxis: { key: 'm1', isMetric: true },
    });
  });
  test('should render a div with a message when the number of unique params/metrics is less than three', () => {
    const props = {
      ...commonProps,
      paramLists: [[{ key: 'p1', value: 1 }], [{ key: 'p1', value: 2 }], [{ key: 'p1', value: 3 }]],
      metricLists: [[{ key: 'm1', value: 4 }], [{ key: 'm1', value: 5 }], [{ key: 'm1', value: 6 }]],
    };
    wrapper = shallow(<CompareRunContour {...props} />);
    expect(wrapper.length).toBe(1);
    expect(wrapper.find('div').length).toBe(1);
    instance = wrapper.instance();
    expect(instance.state).toEqual({ disabled: true });
  });
  test('should remove non-numeric params/metrics', () => {
    const props = {
      ...commonProps,
      paramLists: [
        [
          { key: 'p1', value: 1 },
          { key: 'p2', value: 'a' },
          { key: 'p3', value: 2 },
        ],
        [
          { key: 'p1', value: 3 },
          { key: 'p2', value: 'b' },
          { key: 'p3', value: 3 },
        ],
        [
          { key: 'p1', value: 5 },
          { key: 'p2', value: 'c' },
          { key: 'p3', value: 6 },
        ],
      ],
      metricLists: [[{ key: 'm1', value: 7 }], [{ key: 'm1', value: 8 }], [{ key: 'm1', value: 9 }]],
    };
    wrapper = shallow(<CompareRunContour {...props} />);
    expect(wrapper.length).toBe(1);
    instance = wrapper.instance();
    // 'p2' should be removed because it's a non-numeric parameter.
    expect(instance.paramKeys).toEqual(['p1', 'p3']);
    expect(instance.metricKeys).toEqual(['m1']);
    expect(instance.state).toEqual({
      disabled: false,
      reverseColor: false,
      xaxis: { key: 'p1', isMetric: false },
      yaxis: { key: 'p3', isMetric: false },
      zaxis: { key: 'm1', isMetric: true },
    });
  });
  test('should render with runs without metrics', () => {
    const props = {
      ...commonProps,
      paramLists: [
        [
          { key: 'p1', value: 1 },
          { key: 'p2', value: 2 },
          { key: 'p3', value: 3 },
        ],
        [
          { key: 'p1', value: 4 },
          { key: 'p2', value: 5 },
          { key: 'p3', value: 6 },
        ],
        [
          { key: 'p1', value: 7 },
          { key: 'p2', value: 8 },
          { key: 'p3', value: 9 },
        ],
      ],
      metricLists: [[], [], []],
    };
    wrapper = shallow(<CompareRunContour {...props} />);
    expect(wrapper.length).toBe(1);
    instance = wrapper.instance();
    expect(instance.paramKeys).toEqual(['p1', 'p2', 'p3']);
    expect(instance.metricKeys).toEqual([]);
    expect(instance.state).toEqual({
      disabled: false,
      reverseColor: false,
      xaxis: { key: 'p1', isMetric: false },
      yaxis: { key: 'p2', isMetric: false },
      zaxis: { key: 'p3', isMetric: false },
    });
  });
  test('should render with runs without params', () => {
    const props = {
      ...commonProps,
      paramLists: [[], [], []],
      metricLists: [
        [
          { key: 'm1', value: 1 },
          { key: 'm2', value: 2 },
          { key: 'm3', value: 3 },
        ],
        [
          { key: 'm1', value: 4 },
          { key: 'm2', value: 5 },
          { key: 'm3', value: 6 },
        ],
        [
          { key: 'm1', value: 7 },
          { key: 'm2', value: 8 },
          { key: 'm3', value: 9 },
        ],
      ],
    };
    wrapper = shallow(<CompareRunContour {...props} />);
    expect(wrapper.length).toBe(1);
    instance = wrapper.instance();
    expect(instance.paramKeys).toEqual([]);
    expect(instance.metricKeys).toEqual(['m1', 'm2', 'm3']);
    expect(instance.state).toEqual({
      disabled: false,
      reverseColor: false,
      xaxis: { key: 'm1', isMetric: true },
      yaxis: { key: 'm2', isMetric: true },
      zaxis: { key: 'm3', isMetric: true },
    });
  });
  test('should render when params/metrics corresponding to the x & y axes are only present in some runs', () => {
    const props = {
      ...commonProps,
      paramLists: [
        [
          { key: 'p1', value: 1 },
          { key: 'p2', value: 2 },
        ],
        [
          { key: 'p1', value: 3 },
          { key: 'p2', value: 4 },
        ],
        // this run does not contain 'b'.
        [{ key: 'p1', value: 5 }],
      ],
      metricLists: [[{ key: 'm1', value: 6 }], [{ key: 'm1', value: 7 }], [{ key: 'm1', value: 8 }]],
    };
    wrapper = shallow(<CompareRunContour {...props} />);
    expect(wrapper.length).toBe(1);
    instance = wrapper.instance();
    expect(instance.paramKeys).toEqual(['p1', 'p2']);
    expect(instance.metricKeys).toEqual(['m1']);
    expect(instance.state).toEqual({
      disabled: false,
      reverseColor: false,
      xaxis: { key: 'p1', isMetric: false },
      yaxis: { key: 'p2', isMetric: false },
      zaxis: { key: 'm1', isMetric: true },
    });
  });
  test('should render a warning message when X or Y axis does not have enough data points', () => {
    const props = {
      ...commonProps,
      paramLists: [
        [
          { key: 'p1', value: 0 },
          { key: 'p2', value: 1 },
        ],
        [
          { key: 'p1', value: 0 },
          { key: 'p2', value: 2 },
        ],
        [
          { key: 'p1', value: 0 },
          { key: 'p2', value: 3 },
        ],
      ],
      metricLists: [[{ key: 'm1', value: 4 }], [{ key: 'm1', value: 5 }], [{ key: 'm1', value: 6 }]],
    };
    wrapper = shallowWithIntl(<CompareRunContour {...props} />).dive();
    const renderControlsElement = () => wrapper.dive();
    // X axis: p1 | Y axis: p2
    expect(renderControlsElement().text().includes("The X axis doesn't have enough unique data points")).toBe(true);
    // X axis: p2 | Y axis: p1
    wrapper.setState({
      xaxis: { key: 'p2', isMetric: false },
      yaxis: { key: 'p1', isMetric: false },
    });
    expect(renderControlsElement().text().includes("The Y axis doesn't have enough unique data points")).toBe(true);
    // X axis: p1 | Y axis: p1
    wrapper.setState({ xaxis: { key: 'p1', isMetric: false } });
    expect(renderControlsElement().text().includes("The X and Y axes don't have enough unique data points")).toBe(true);
    // X axis: p2 | Y axis: p2
    wrapper.setState({
      xaxis: { key: 'p2', isMetric: false },
      yaxis: { key: 'p2', isMetric: false },
    });
    expect(renderControlsElement().text().includes('have enough unique data points')).toBe(false);
  });
});
