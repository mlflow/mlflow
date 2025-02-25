import { render, screen, fireEvent, act } from '@testing-library/react';
import { RunsChartsConfigureLineChart } from './RunsChartsConfigureLineChart';
import { IntlProvider } from 'react-intl';
import { RunsChartsLineChartXAxisType } from '../RunsCharts.common';
import { DesignSystemProvider } from '@databricks/design-system';

describe('RunsChartsConfigureLineChart', () => {
  test('should update x and y range when manual controls are changed', () => {
    // Arrange
    const metricKeyList = ['metric1'];
    const onStateChangeMock = jest.fn();

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <RunsChartsConfigureLineChart
            metricKeyList={metricKeyList}
            state={{
              metricKey: 'metric1',
              lineSmoothness: 0,
              xAxisScaleType: 'linear',
              scaleType: 'linear',
              xAxisKey: RunsChartsLineChartXAxisType.STEP,
              range: {
                xMin: 0,
                xMax: 10,
                yMin: 0,
                yMax: 10,
              },
              selectedMetricKeys: ['metric1'],
            }}
            onStateChange={onStateChangeMock}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const xRangeMinInput = screen.getByLabelText('x-axis-min');
    const xRangeMaxInput = screen.getByLabelText('x-axis-max');
    const yRangeMinInput = screen.getByLabelText('y-axis-min');
    const yRangeMaxInput = screen.getByLabelText('y-axis-max');

    fireEvent.change(xRangeMinInput, { target: { value: '10' } });
    expect(onStateChangeMock.mock.calls[0][0]({})).toEqual({
      range: {
        xMin: 10,
        xMax: 10,
      },
    });
    fireEvent.change(xRangeMaxInput, { target: { value: '40' } });
    expect(onStateChangeMock.mock.lastCall[0]({})).toEqual({
      range: {
        xMin: 10,
        xMax: 40,
      },
    });
    fireEvent.change(yRangeMinInput, { target: { value: '5' } });
    expect(onStateChangeMock.mock.lastCall[0]({})).toEqual({
      range: {
        yMin: 5,
        yMax: 10,
      },
    });
    fireEvent.change(yRangeMaxInput, { target: { value: '15' } });
    expect(onStateChangeMock.mock.lastCall[0]({})).toEqual({
      range: {
        yMin: 5,
        yMax: 15,
      },
    });
  });

  test('correctly transitions state of x and y range values', () => {
    // Arrange
    const metricKeyList = ['metric1'];
    const onStateChangeMock = jest.fn();

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <RunsChartsConfigureLineChart
            metricKeyList={metricKeyList}
            state={{
              metricKey: 'metric1',
              lineSmoothness: 0,
              xAxisScaleType: 'linear',
              scaleType: 'linear',
              xAxisKey: RunsChartsLineChartXAxisType.STEP,
              range: {
                xMin: undefined,
                xMax: undefined,
                yMin: undefined,
                yMax: undefined,
              },
              selectedMetricKeys: ['metric1'],
            }}
            onStateChange={onStateChangeMock}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const xRangeMinInput = screen.getByLabelText('x-axis-min');
    const xRangeMaxInput = screen.getByLabelText('x-axis-max');
    const yRangeMinInput = screen.getByLabelText('y-axis-min');
    const yRangeMaxInput = screen.getByLabelText('y-axis-max');

    fireEvent.change(xRangeMinInput, { target: { value: '10' } });
    expect(onStateChangeMock).toHaveBeenCalledTimes(0);
    fireEvent.change(xRangeMaxInput, { target: { value: '40' } });
    expect(onStateChangeMock).toHaveBeenCalledTimes(1);
    fireEvent.change(yRangeMinInput, { target: { value: '5' } });
    expect(onStateChangeMock).toHaveBeenCalledTimes(1);
    fireEvent.change(yRangeMaxInput, { target: { value: '15' } });
    expect(onStateChangeMock).toHaveBeenCalledTimes(2);
    fireEvent.change(yRangeMaxInput, { target: { value: '' } });
    expect(onStateChangeMock).toHaveBeenCalledTimes(2);
    fireEvent.change(yRangeMinInput, { target: { value: '' } });
    expect(onStateChangeMock).toHaveBeenCalledTimes(3);
  });

  test('correctly transitions x and y ranges when switching to log value', () => {
    // Arrange
    const metricKeyList = ['metric1'];
    const onStateChangeMock = jest.fn();

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <RunsChartsConfigureLineChart
            metricKeyList={metricKeyList}
            state={{
              metricKey: 'metric1',
              lineSmoothness: 0,
              xAxisScaleType: 'linear',
              scaleType: 'linear',
              xAxisKey: RunsChartsLineChartXAxisType.STEP,
              range: {
                xMin: -10,
                xMax: 10,
                yMin: -10,
                yMax: -5,
              },
              selectedMetricKeys: ['metric1'],
            }}
            onStateChange={onStateChangeMock}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const yScaleTypeSelect = screen.getByLabelText('y-axis-log');
    const xScaleTypeSelect = screen.getByLabelText('x-axis-log');

    fireEvent.click(xScaleTypeSelect);
    expect(onStateChangeMock.mock.lastCall[0]({})).toEqual({
      range: {
        xMin: Math.log10(1),
        xMax: Math.log10(10),
      },
      xAxisScaleType: 'log',
    });

    // Should reset when both range values are invalid log values
    fireEvent.click(yScaleTypeSelect);
    expect(onStateChangeMock.mock.lastCall[0]({})).toEqual({
      range: {
        yMin: undefined,
        yMax: undefined,
      },
      scaleType: 'log',
    });
  });

  test('correctly transitions x and y ranges when switching from log value', () => {
    // Arrange
    const metricKeyList = ['metric1'];
    const onStateChangeMock = jest.fn();

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <RunsChartsConfigureLineChart
            metricKeyList={metricKeyList}
            state={{
              metricKey: 'metric1',
              lineSmoothness: 0,
              xAxisScaleType: 'log',
              scaleType: 'log',
              xAxisKey: RunsChartsLineChartXAxisType.STEP,
              range: {
                xMin: 1,
                xMax: 4,
                yMin: -1,
                yMax: 1,
              },
              selectedMetricKeys: ['metric1'],
            }}
            onStateChange={onStateChangeMock}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const yScaleTypeSelect = screen.getByLabelText('y-axis-log');
    const xScaleTypeSelect = screen.getByLabelText('x-axis-log');

    fireEvent.click(xScaleTypeSelect);
    expect(onStateChangeMock.mock.lastCall[0]({})).toEqual({
      range: {
        xMin: 10,
        xMax: 10000,
      },
      xAxisScaleType: 'linear',
    });

    // Should reset when both range values are invalid log values
    fireEvent.click(yScaleTypeSelect);
    expect(onStateChangeMock.mock.lastCall[0]({})).toEqual({
      range: {
        yMin: 0.1,
        yMax: 10,
      },
      scaleType: 'linear',
    });
  });
});
