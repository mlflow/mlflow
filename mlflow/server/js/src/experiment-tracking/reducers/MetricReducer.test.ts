/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import {
  metricsByKey,
  getLatestMetrics,
  getMinMetrics,
  getMaxMetrics,
  getMetricsByKey,
  metricsByRunUuid,
  minMetricsByRunUuid,
  maxMetricsByRunUuid,
} from './MetricReducer';
import { GET_METRIC_HISTORY_API, GET_METRIC_HISTORY_API_BULK } from '../actions';
import { Metric } from '../sdk/MlflowMessages';
import { fulfilled } from '../../common/utils/ActionUtils';

describe('test getMetricsByKey', () => {
  test('run has no keys', () => {
    const state = {
      entities: {
        metricsByRunUuid: {
          run1: {},
        },
      },
    };
    expect(getMetricsByKey('run1', undefined, state)).toEqual(undefined);
  });

  test('run has no matching key', () => {
    const state = {
      entities: {
        metricsByRunUuid: {
          run1: { m1: undefined },
        },
      },
    };
    expect(getMetricsByKey('run1', 'm2', state)).toEqual(undefined);
  });

  test('returns matching key (1)', () => {
    const state = {
      entities: {
        metricsByRunUuid: {
          run1: { m1: 1, m2: undefined },
        },
      },
    };
    expect(getMetricsByKey('run1', 'm2', state)).toEqual(undefined);
  });

  test('returns matching key (2)', () => {
    const state = {
      entities: {
        metricsByRunUuid: {
          run1: { m1: 1, m2: 2 },
        },
      },
    };
    expect(getMetricsByKey('run1', 'm2', state)).toEqual(2);
  });
});

describe('test getLatestMetrics', () => {
  test('handle empty state', () => {
    const state = {
      entities: {
        latestMetricsByRunUuid: {},
      },
    };
    expect(getLatestMetrics('run1', state)).toEqual(undefined);
  });

  test('missing run', () => {
    const state = {
      entities: {
        latestMetricsByRunUuid: {
          run2: [],
        },
      },
    };
    expect(getLatestMetrics('run1', state)).toEqual(undefined);
  });

  test('run has no metrics', () => {
    const state = {
      entities: {
        latestMetricsByRunUuid: {
          run1: [],
        },
      },
    };
    expect(getLatestMetrics('run1', state)).toEqual([]);
  });

  test('returns all metrics', () => {
    const state = {
      entities: {
        latestMetricsByRunUuid: {
          run1: { m1: 1, m2: 2 },
        },
      },
    };
    expect(getLatestMetrics('run1', state)).toEqual({ m1: 1, m2: 2 });
  });

  test('returns correct metrics', () => {
    const state = {
      entities: {
        latestMetricsByRunUuid: {
          run1: { m1: 1, m2: 2 },
          run2: { m1: 3, m2: 4 },
        },
      },
    };
    expect(getLatestMetrics('run2', state)).toEqual({ m1: 3, m2: 4 });
  });
});

describe('test getMinMetrics', () => {
  test('handle empty state', () => {
    const state = {
      entities: {
        minMetricsByRunUuid: {},
      },
    };
    expect(getMinMetrics('run1', state)).toEqual(undefined);
  });

  test('returns correct metrics', () => {
    const state = {
      entities: {
        minMetricsByRunUuid: {
          run1: { m1: 1, m2: 2 },
          run2: { m1: 3, m2: 4 },
        },
      },
    };
    expect(getMinMetrics('run2', state)).toEqual({ m1: 3, m2: 4 });
  });
});

describe('test getMaxMetrics', () => {
  test('handle empty state', () => {
    const state = {
      entities: {
        maxMetricsByRunUuid: {},
      },
    };
    expect(getMaxMetrics('run1', state)).toEqual(undefined);
  });

  test('returns correct metrics', () => {
    const state = {
      entities: {
        maxMetricsByRunUuid: {
          run1: { m1: 1, m2: 2 },
          run2: { m1: 3, m2: 4 },
        },
      },
    };
    expect(getMaxMetrics('run2', state)).toEqual({ m1: 3, m2: 4 });
  });
});

const mockMetric = (key: any, value: any, timestamp = 1234567890000, step = 1, run_id = undefined) => {
  const metric = {
    key: key,
    value: value,
    timestamp: timestamp,
    step: step,
    run_id: run_id,
  };
  return [metric, (Metric as any).fromJs(metric)];
};

describe('test metricsByKey', () => {
  // All tests for GET_METRIC_HISTORY_API are valid for GET_METRIC_HISTORY_API_BULK
  // due to the same treatment inside metricsByKey()
  test('intial state (1)', () => {
    expect(metricsByKey({}, {}, undefined)).toEqual({});
  });

  test('intial state (2)', () => {
    expect(metricsByKey({ 1: '1' }, {}, undefined)).toEqual({ 1: '1' });
  });

  test('GET_METRIC_HISTORY_API handles empty state and empty metrics', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { key: 'm1' },
    };
    const metrics: any = [];
    expect(metricsByKey({}, action, metrics)).toEqual({ m1: [] });
  });

  test('GET_METRIC_HISTORY_API handles empty state with valid metrics', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { key: 'm1' },
    };
    const [m1, m1proto] = mockMetric('m1', 5);
    const metrics = [m1];
    expect(metricsByKey({}, action, metrics)).toEqual({ m1: [m1proto] });
  });

  test('GET_METRIC_HISTORY_API updates stale metrics', () => {
    const [, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 6);
    const state = {
      acc: [m1proto],
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { key: 'acc' },
    };
    const metrics = [m2];
    expect(metricsByKey(state, action, metrics)).toEqual({ acc: [m2proto] });
  });

  test('GET_METRIC_HISTORY_API refreshes metrics', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 6);
    const state = {
      acc: [m1proto],
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { key: 'acc' },
    };
    const metrics = [m1, m2];
    expect(metricsByKey(state, action, metrics)).toEqual({ acc: [m1proto, m2proto] });
  });

  test('GET_METRIC_HISTORY_API leaves other metric keys in state unaffected', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 6);
    const [, m3proto] = mockMetric('loss', 0.001);
    const state = {
      loss: [m3proto],
      acc: [m1proto],
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { key: 'acc' },
    };
    const metrics = [m1, m2];
    expect(metricsByKey(state, action, metrics)).toEqual({
      acc: [m1proto, m2proto],
      loss: [m3proto],
    });
  });
});

describe('test metricsByRunUuid', () => {
  test('initial state', () => {
    expect(metricsByRunUuid({}, {})).toEqual({});
  });

  test('state returned as is for empty action', () => {
    expect(metricsByRunUuid({ a: 1 }, {})).toEqual({ a: 1 });
  });

  test('GET_METRIC_HISTORY_API handles empty state', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'm1' },
      payload: {},
    };
    expect(metricsByRunUuid({}, action)).toEqual({ run1: { m1: [] } });
  });

  test('GET_METRIC_HISTORY_API_BULK handles empty state', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API_BULK),
      meta: { runUuids: ['run1', 'run2'], key: 'm1' },
      payload: {},
    };
    expect(metricsByRunUuid({}, action)).toEqual({ run1: { m1: [] }, run2: { m1: [] } });
  });

  test('GET_METRIC_HISTORY_API handles missing run in state', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'm1' },
      payload: {},
    };
    const state = { run2: undefined };
    expect(metricsByRunUuid(state, action)).toEqual({ run1: { m1: [] }, run2: undefined });
  });

  test('GET_METRIC_HISTORY_API_BULK handles missing run in state', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API_BULK),
      meta: { runUuids: ['run1', 'run2'], key: 'm1' },
      payload: {},
    };
    const state = { run3: undefined };
    expect(metricsByRunUuid(state, action)).toEqual({
      run1: { m1: [] },
      run2: { m1: [] },
      run3: undefined,
    });
  });

  test('GET_METRIC_HISTORY_API returns appropriate metrics', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 6);
    const state = {
      run1: {
        acc: [m1proto],
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(metricsByRunUuid(state, action)).toEqual({ run1: { acc: [m1proto, m2proto] } });
  });

  test('GET_METRIC_HISTORY_API_BULK returns appropriate metrics and separates them between runs', () => {
    // @ts-expect-error TS(2345): Argument of type '"run1"' is not assignable to par... Remove this comment to see the full error message
    const [m1, m1proto] = mockMetric('acc', 5, undefined, undefined, 'run1');
    // @ts-expect-error TS(2345): Argument of type '"run2"' is not assignable to par... Remove this comment to see the full error message
    const [m2, m2proto] = mockMetric('acc', 6, undefined, undefined, 'run2');
    const state = {
      run1: {
        acc: [m1proto],
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API_BULK),
      meta: { runUuids: ['run1', 'run2'], key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(metricsByRunUuid(state, action)).toEqual({
      run1: { acc: [m1proto] },
      run2: { acc: [m2proto] },
    });
  });

  test('GET_METRIC_HISTORY_API updates state for relevant metric and leaves other metrics unaffected', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 6);
    const [, m3proto] = mockMetric('loss', 0.001);
    const state = {
      run1: {
        acc: [m1proto],
        loss: [m3proto],
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(metricsByRunUuid(state, action)).toEqual({
      run1: {
        acc: [m1proto, m2proto],
        loss: [m3proto],
      },
    });
  });

  test("GET_METRIC_HISTORY_API updates state for relevant run's metrics and leaves other runs unaffected", () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 6);
    const [, m3proto] = mockMetric('loss', 0.001);
    const state = {
      run1: {
        acc: [m1proto],
      },
      run2: {
        acc: [m3proto],
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(metricsByRunUuid(state, action)).toEqual({
      run1: {
        acc: [m1proto, m2proto],
      },
      run2: {
        acc: [m3proto],
      },
    });
  });

  test("GET_METRIC_HISTORY_API_BULK updates state for relevant runs' metrics and leaves other runs unaffected", () => {
    // @ts-expect-error TS(2345): Argument of type '"run1"' is not assignable to par... Remove this comment to see the full error message
    const [, m1proto] = mockMetric('acc', 5, 1, 1, 'run1');
    // @ts-expect-error TS(2345): Argument of type '"run2"' is not assignable to par... Remove this comment to see the full error message
    const [, m2proto] = mockMetric('acc', 6, 1, 1, 'run2');
    // @ts-expect-error TS(2345): Argument of type '"run3"' is not assignable to par... Remove this comment to see the full error message
    const [m3, m3proto] = mockMetric('acc', 7, 1, 1, 'run3');
    const state = {
      run1: {
        acc: [m1proto],
      },
      run2: {
        acc: [m2proto],
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API_BULK),
      meta: { runUuids: ['run3'], key: 'acc' },
      payload: {
        metrics: [m3],
      },
    };
    expect(metricsByRunUuid(state, action)).toEqual({
      run1: {
        acc: [m1proto],
      },
      run2: {
        acc: [m2proto],
      },
      run3: {
        acc: [m3proto],
      },
    });
  });
});

describe('test minMetricsByRunUuid', () => {
  test('initial state', () => {
    expect(minMetricsByRunUuid({}, {})).toEqual({});
  });

  test('state returned as is for empty action', () => {
    expect(minMetricsByRunUuid({ a: 1 }, {})).toEqual({ a: 1 });
  });

  test('GET_METRIC_HISTORY_API handles empty state', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'm1' },
      payload: {},
    };
    expect(minMetricsByRunUuid({}, action)).toEqual({});
  });

  test('GET_METRIC_HISTORY_API handles empty metrics', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'm1' },
      payload: {
        metrics: [],
      },
    };
    expect(minMetricsByRunUuid({}, action)).toEqual({});
  });

  test('GET_METRIC_HISTORY_API returns correct minimum metrics', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2] = mockMetric('acc', 6);
    const [m3, m3proto] = mockMetric('acc', 4);
    const [m4] = mockMetric('acc', 8);
    const [m5] = mockMetric('acc', NaN);
    const state = {
      run1: {
        acc: m1proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2, m3, m4, m5],
      },
    };
    expect(minMetricsByRunUuid(state, action)).toEqual({ run1: { acc: m3proto } });
  });

  test('GET_METRIC_HISTORY_API_BULK returns correct minimum metrics', () => {
    // @ts-expect-error TS(2345): Argument of type '"run1"' is not assignable to par... Remove this comment to see the full error message
    const [m1, m1proto] = mockMetric('acc', 5, undefined, undefined, 'run1');
    // @ts-expect-error TS(2345): Argument of type '"run1"' is not assignable to par... Remove this comment to see the full error message
    const [m2, m2proto] = mockMetric('acc', 4, undefined, undefined, 'run1');
    // @ts-expect-error TS(2345): Argument of type '"run2"' is not assignable to par... Remove this comment to see the full error message
    const [m3, m3proto] = mockMetric('acc', 1, undefined, undefined, 'run2');
    // @ts-expect-error TS(2345): Argument of type '"run2"' is not assignable to par... Remove this comment to see the full error message
    const [m4] = mockMetric('acc', 2, undefined, undefined, 'run2');
    const state = {
      run1: {
        acc: m1proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API_BULK),
      meta: { runUuids: ['run1', 'run2'], key: 'acc' },
      payload: {
        metrics: [m1, m2, m3, m4],
      },
    };
    expect(minMetricsByRunUuid(state, action)).toEqual({
      run1: { acc: m2proto },
      run2: { acc: m3proto },
    });
  });

  test('GET_METRIC_HISTORY_API updates state for relevant metric and leaves other metrics unaffected', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 4);
    const [, m3proto] = mockMetric('loss', 6);
    const state = {
      run1: {
        acc: m1proto,
        loss: m3proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(minMetricsByRunUuid(state, action)).toEqual({
      run1: { acc: m2proto, loss: m3proto },
    });
  });

  test('GET_METRIC_HISTORY_API updates state for relevant run and leaves other runs unaffected', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 4);
    const [, m3proto] = mockMetric('acc', 6);
    const state = {
      run1: {
        acc: m1proto,
      },
      run2: {
        acc: m3proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(minMetricsByRunUuid(state, action)).toEqual({
      run1: { acc: m2proto },
      run2: { acc: m3proto },
    });
  });
});

describe('test maxMetricsByRunUuid', () => {
  test('initial state', () => {
    expect(maxMetricsByRunUuid({}, {})).toEqual({});
  });

  test('state returned as is for empty action', () => {
    expect(maxMetricsByRunUuid({ a: 1 }, {})).toEqual({ a: 1 });
  });

  test('GET_METRIC_HISTORY_API handles empty state', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'm1' },
      payload: {},
    };
    expect(maxMetricsByRunUuid({}, action)).toEqual({});
  });

  test('GET_METRIC_HISTORY_API handles empty metrics', () => {
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'm1' },
      payload: {
        metrics: [],
      },
    };
    expect(maxMetricsByRunUuid({}, action)).toEqual({});
  });

  test('GET_METRIC_HISTORY_API returns correct maximum metrics', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2] = mockMetric('acc', 6);
    const [m3, m3proto] = mockMetric('acc', 9);
    const [m4] = mockMetric('acc', 8);
    const [m5] = mockMetric('acc', NaN);
    const state = {
      run1: {
        acc: m1proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2, m3, m4, m5],
      },
    };
    expect(maxMetricsByRunUuid(state, action)).toEqual({ run1: { acc: m3proto } });
  });

  test('GET_METRIC_HISTORY_API_BULK returns correct maximum metrics', () => {
    // @ts-expect-error TS(2345): Argument of type '"run1"' is not assignable to par... Remove this comment to see the full error message
    const [m1, m1proto] = mockMetric('acc', 12, undefined, undefined, 'run1');
    // @ts-expect-error TS(2345): Argument of type '"run1"' is not assignable to par... Remove this comment to see the full error message
    const [m2] = mockMetric('acc', 8, undefined, undefined, 'run1');
    // @ts-expect-error TS(2345): Argument of type '"run2"' is not assignable to par... Remove this comment to see the full error message
    const [m3] = mockMetric('acc', 9, undefined, undefined, 'run2');
    // @ts-expect-error TS(2345): Argument of type '"run2"' is not assignable to par... Remove this comment to see the full error message
    const [m4, m4proto] = mockMetric('acc', 11, undefined, undefined, 'run2');
    const state = {
      run1: {
        acc: m1proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API_BULK),
      meta: { runUuids: ['run1', 'run2'], key: 'acc' },
      payload: {
        metrics: [m1, m2, m3, m4],
      },
    };
    expect(maxMetricsByRunUuid(state, action)).toEqual({
      run1: { acc: m1proto },
      run2: { acc: m4proto },
    });
  });

  test('GET_METRIC_HISTORY_API updates state for relevant metric and leaves other metrics unaffected', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 7);
    const [, m3proto] = mockMetric('loss', 6);
    const state = {
      run1: {
        acc: m1proto,
        loss: m3proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(maxMetricsByRunUuid(state, action)).toEqual({
      run1: { acc: m2proto, loss: m3proto },
    });
  });

  test('GET_METRIC_HISTORY_API updates state for relevant run and leaves other runs unaffected', () => {
    const [m1, m1proto] = mockMetric('acc', 5);
    const [m2, m2proto] = mockMetric('acc', 7);
    const [, m3proto] = mockMetric('acc', 6);
    const state = {
      run1: {
        acc: m1proto,
      },
      run2: {
        acc: m3proto,
      },
    };
    const action = {
      type: fulfilled(GET_METRIC_HISTORY_API),
      meta: { runUuid: 'run1', key: 'acc' },
      payload: {
        metrics: [m1, m2],
      },
    };
    expect(maxMetricsByRunUuid(state, action)).toEqual({
      run1: { acc: m2proto },
      run2: { acc: m3proto },
    });
  });
});
