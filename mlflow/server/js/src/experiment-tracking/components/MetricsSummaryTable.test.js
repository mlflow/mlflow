import React from 'react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import MetricsSummaryTable from './MetricsSummaryTable';
import { HtmlTableView } from './HtmlTableView';
import { Metric } from '../sdk/MlflowMessages';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('MetricsSummaryTable', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {
      runUuids: ['uuid-1234-5678-9012'],
      runDisplayNames: ['run 0'],
      metricKeys: ['train_loss'],
    };

    minimalStore = mockStore({
      entities: {
        latestMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
        minMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
        maxMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
      },
    });
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <MetricsSummaryTable {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(MetricsSummaryTable).length).toBe(1);
  });

  test('should render metric summary values', () => {
    const store = mockStore({
      entities: {
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': { train_loss: Metric.fromJs({ key: 'train_loss', value: 1 }) },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': { train_loss: Metric.fromJs({ key: 'train_loss', value: 2 }) },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': { train_loss: Metric.fromJs({ key: 'train_loss', value: 3 }) },
        },
      },
    });
    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <MetricsSummaryTable {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );

    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(1);

    const { columns, values } = table.props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(1);

    const html = wrapper.html();
    expect(html).toContain('train_loss');
    expect(html).not.toContain('run 0');

    for (let i = 1; i < 4; ++i) {
      expect(html).toContain(i.toString());
    }
  });

  test('should render multiple metric values for single run', () => {
    const store = mockStore({
      entities: {
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 1 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 2 }),
          },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 3 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 4 }),
          },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 5 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 6 }),
          },
        },
      },
    });
    const props = {
      ...minimalProps,
      metricKeys: ['train_loss', 'val_loss'],
    };
    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <MetricsSummaryTable {...props} />
        </BrowserRouter>
      </Provider>,
    );

    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(1);

    const { columns, values } = table.props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);

    const html = wrapper.html();
    expect(html).toContain('train_loss');
    expect(html).toContain('val_loss');
    expect(html).not.toContain('run 0');

    for (let i = 1; i < 7; ++i) {
      expect(html).toContain(i.toString());
    }
  });

  test('should render single metric for multiple runs', () => {
    const store = mockStore({
      entities: {
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': { train_loss: Metric.fromJs({ key: 'train_loss', value: 1 }) },
          'uuid-1234-5678-9013': { train_loss: Metric.fromJs({ key: 'train_loss', value: 2 }) },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': { train_loss: Metric.fromJs({ key: 'train_loss', value: 3 }) },
          'uuid-1234-5678-9013': { train_loss: Metric.fromJs({ key: 'train_loss', value: 4 }) },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': { train_loss: Metric.fromJs({ key: 'train_loss', value: 5 }) },
          'uuid-1234-5678-9013': { train_loss: Metric.fromJs({ key: 'train_loss', value: 6 }) },
        },
      },
    });
    const props = {
      runUuids: ['uuid-1234-5678-9012', 'uuid-1234-5678-9013'],
      runDisplayNames: ['run 0', 'run 1'],
      metricKeys: ['train_loss'],
    };
    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <MetricsSummaryTable {...props} />
        </BrowserRouter>
      </Provider>,
    );

    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(1);

    const { columns, values } = table.props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);

    const html = wrapper.html();
    expect(html).not.toContain('train_loss');
    expect(html).toContain('run 0');
    expect(html).toContain('run 1');

    for (let i = 1; i < 7; ++i) {
      expect(html).toContain(i.toString());
    }
  });

  test('should render multiple metrics for multiple runs', () => {
    const store = mockStore({
      entities: {
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 1 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 2 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 3 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 4 }),
          },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 5 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 6 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 7 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 8 }),
          },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 9 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 10 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: Metric.fromJs({ key: 'train_loss', value: 11 }),
            val_loss: Metric.fromJs({ key: 'val_loss', value: 12 }),
          },
        },
      },
    });
    const props = {
      runUuids: ['uuid-1234-5678-9012', 'uuid-1234-5678-9013'],
      runDisplayNames: ['run 0', 'run 1'],
      metricKeys: ['train_loss', 'val_loss'],
    };
    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <MetricsSummaryTable {...props} />
        </BrowserRouter>
      </Provider>,
    );

    const tables = wrapper.find(HtmlTableView);
    expect(tables.length).toBe(2);

    let { columns, values } = tables.at(0).props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);

    ({ columns, values } = tables.at(1).props());
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);

    const html = wrapper.html();
    expect(html).toContain('train_loss');
    expect(html).toContain('val_loss');
    expect(html).toContain('run 0');
    expect(html).toContain('run 1');

    for (let i = 1; i < 13; ++i) {
      expect(html).toContain(i.toString());
    }
  });
});
