import React, { useEffect, useRef } from 'react';
import { act, fireEvent, renderWithIntl, screen } from '../../../../common/utils/TestUtils.react18';
import { RunsChartsLineChartXAxisType } from '../components/RunsCharts.common';
import { useRunsMultipleTracesTooltipData } from './useRunsChartsMultipleTracesTooltip';
import type { Figure } from 'react-plotly.js';
import invariant from 'invariant';
import type { RunsCompareMultipleTracesTooltipData } from '../components/RunsMetricsLinePlot';

const testFigure: Figure = {
  layout: {
    xaxis: {
      range: [0, 100],
    },
  },
  data: [],
  frames: null,
};

jest.useFakeTimers();

const onHoverTestCallback = jest.fn();
const onUnhoverTestCallback = jest.fn();

describe('useCompareRunsAllTracesTooltipData', () => {
  const getLastHoverCallbackData = (): RunsCompareMultipleTracesTooltipData =>
    jest.mocked(onHoverTestCallback).mock?.lastCall?.[2];

  const hoverPointerOnClientX = (clientX: number) => {
    jest.advanceTimersByTime(200);

    fireEvent(
      screen.getByTestId('draglayer'),
      new MouseEvent('pointermove', {
        clientX,
      }),
    );
  };

  const renderTestComponent = () => {
    let onPointHover: ({ points }: any) => void = () => {};
    const TestComponent = () => {
      const testChartElementRef = useRef<HTMLDivElement>(null);

      const hookResult = useRunsMultipleTracesTooltipData({
        onUnhover: onUnhoverTestCallback,
        onHover: onHoverTestCallback,
        disabled: false,
        legendLabelData: [
          { color: 'red', label: 'First trace', uuid: 'first-trace', metricKey: 'metric_a' },
          { color: 'blue', label: 'Second trace', uuid: 'second-trace', metricKey: 'metric_a' },
        ],
        plotData: [
          {
            x: [10, 20, 30],
            y: [110, 220, 330],
            uuid: 'first-trace',
            metricKey: 'metric_a',
          },
          {
            x: [10, 40, 60],
            y: [70, 440, 660],
            uuid: 'second-trace',
            metricKey: 'metric_a',
          },
        ],
        runsData: [
          { displayName: 'First trace', uuid: 'first-trace' },
          { displayName: 'Second trace', uuid: 'second-trace' },
        ],
        xAxisKeyLabel: 'Step',
        containsMultipleMetricKeys: false,
        xAxisKey: RunsChartsLineChartXAxisType.STEP,
        setHoveredPointIndex: () => {},
      });

      const { initHandler, scanlineElement } = hookResult;
      onPointHover = hookResult.onPointHover;

      useEffect(() => {
        if (!testChartElementRef.current) {
          return;
        }
        const SVG = testChartElementRef.current.querySelector('.main-svg');
        invariant(SVG, 'SVG should exist');
        const draglayer = SVG.querySelector('.nsewdrag');
        invariant(draglayer, 'draglayer should exist');
        SVG.getBoundingClientRect = jest.fn<any, any>(() => ({
          width: 200,
          x: 0,
        }));
        draglayer.getBoundingClientRect = jest.fn<any, any>(() => ({
          width: 200,
          x: 0,
        }));
        initHandler(testFigure, testChartElementRef.current);
      }, [initHandler]);

      return (
        // Render mock chart element
        <div ref={testChartElementRef}>
          <svg className="main-svg" role="figure">
            <g className="draglayer">
              <g className="nsewdrag" data-testid="draglayer" />
            </g>
          </svg>
          {scanlineElement && React.cloneElement(scanlineElement, { 'data-testid': 'scanline' })}
        </div>
      );
    };
    renderWithIntl(<TestComponent />);

    return { onHoverMock: onHoverTestCallback, onUnhoverMock: onUnhoverTestCallback, onPointHover };
  };
  test('displays multiple traces tooltip', async () => {
    const { onPointHover } = renderTestComponent();

    // First, hover over the chart on exact position containing points of both traces
    hoverPointerOnClientX(10);
    expect(getLastHoverCallbackData().xValue).toEqual(10);
    expect(getLastHoverCallbackData().tooltipLegendItems).toEqual([
      expect.objectContaining({ displayName: 'First trace', uuid: 'first-trace.metric_a', value: 110 }),
      expect.objectContaining({ displayName: 'Second trace', uuid: 'second-trace.metric_a', value: 70 }),
    ]);
    expect(window.getComputedStyle(screen.getByTestId('scanline')).left).toEqual('20px');

    // First, hover over the chart just near the place where only the first trace has a point
    hoverPointerOnClientX(42);
    expect(getLastHoverCallbackData().xValue).toEqual(20);
    expect(getLastHoverCallbackData().tooltipLegendItems).toEqual([
      expect.objectContaining({ displayName: 'First trace', uuid: 'first-trace.metric_a', value: 220 }),
    ]);
    expect(window.getComputedStyle(screen.getByTestId('scanline')).left).toEqual('40px');

    // First, hover over the chart just near the place where only the second trace has a point,
    // also hover over a particular points in chart
    onPointHover({ points: [{ x: 40, y: 440, data: { uuid: 'second-trace.metric_a' } }] });
    hoverPointerOnClientX(77);
    expect(getLastHoverCallbackData().xValue).toEqual(40);
    expect(getLastHoverCallbackData().tooltipLegendItems).toEqual([
      expect.objectContaining({ displayName: 'Second trace', uuid: 'second-trace.metric_a', value: 440 }),
    ]);
    // Hook also reports hovered point
    expect(getLastHoverCallbackData().hoveredDataPoint?.traceUuid).toEqual('second-trace.metric_a');
    expect(window.getComputedStyle(screen.getByTestId('scanline')).left).toEqual('80px');

    // Simulate mouse leaving the chart
    expect(onUnhoverTestCallback).toHaveBeenCalledTimes(0);
    fireEvent(screen.getByTestId('draglayer'), new MouseEvent('pointerleave'));
    expect(onUnhoverTestCallback).toHaveBeenCalledTimes(1);
  });
});
