import { jest, describe, test, expect } from '@jest/globals';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import * as Recharts from 'recharts';
import {
  EvalRunsRecentRunsChart,
  formatEvalRunsXAxisTick,
  getEvalRunsYAxisDomain,
  getEvalRunsBarTooltipPosition,
} from './EvalRunsRecentRunsChart';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { getDefaultChartMetricKeys } from './getDefaultChartMetricKeys';
import type { RunEntity } from '../../../types';

const rechartsSpies = Recharts as unknown as {
  Line: (props: unknown) => unknown;
  YAxis: (props: unknown) => unknown;
  Legend: (props: unknown) => unknown;
};

const makeRun = (
  runUuid: string,
  {
    startTime = 0,
    datasetName,
    metrics = [],
  }: { startTime?: number; datasetName?: string; metrics?: [string, number][] },
): RunEntity => ({
  data: {
    params: [],
    tags: [],
    metrics: metrics.map(([key, value]) => ({
      key,
      value,
      step: 0,
      timestamp: 0,
    })),
  },
  info: {
    artifactUri: '',
    endTime: 0,
    experimentId: 'exp-1',
    lifecycleStage: '',
    runUuid,
    runName: runUuid,
    startTime,
    status: 'FINISHED',
  },
  inputs: datasetName
    ? {
        datasetInputs: [
          {
            dataset: {
              name: datasetName,
              digest: 'd',
              profile: '',
              schema: '',
              source: '',
              sourceType: '',
            },
            tags: [],
          },
        ],
      }
    : undefined,
});

const chartTree = (props: Parameters<typeof EvalRunsRecentRunsChart>[0]) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <MockedReduxStoreProvider
        state={{
          entities: {
            colorByRunUuid: { 'run-a': '#123456', 'run-b': '#654321' },
          },
        }}
      >
        <EvalRunsRecentRunsChart {...props} />
      </MockedReduxStoreProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

const renderChart = (props: Partial<Parameters<typeof EvalRunsRecentRunsChart>[0]> = {}) => {
  const defaultProps: Parameters<typeof EvalRunsRecentRunsChart>[0] = {
    runs: [],
    metricKeys: ['correctness'],
    selectedMetricKeys: ['correctness'],
    onSelectedMetricKeysChange: jest.fn(),
  };
  return render(chartTree({ ...defaultProps, ...props }));
};

const makeBoundingRect = (x: number, y: number, width: number, height: number): DOMRect => ({
  bottom: y + height,
  height,
  left: x,
  right: x + width,
  top: y,
  width,
  x,
  y,
  toJSON: () => ({}),
});

// @ui-test-skill-generated
describe('EvalRunsRecentRunsChart', () => {
  test('uses a compact layout without the redundant dataset picker', () => {
    renderChart({
      compact: true,
      runs: [
        makeRun('run-a', {
          datasetName: 'dataset-a',
          metrics: [['correctness', 0.9]],
        }),
      ],
    });

    expect(screen.getByTestId('eval-runs-chart-area')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /dataset-a/i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'All assessments' })).toBeInTheDocument();
  });

  test('defaults to three assessments in compact mode', () => {
    const metrics: [string, number][] = [
      ['a', 0.1],
      ['b', 0.2],
      ['c', 0.3],
      ['d', 0.4],
    ];
    renderChart({
      compact: true,
      runs: [makeRun('run-a', { metrics })],
      metricKeys: metrics.map(([key]) => key),
      selectedMetricKeys: undefined,
    });

    expect(screen.getByRole('button', { name: '3 assessments' })).toBeInTheDocument();
    expect(screen.getByTestId('line-a')).toBeInTheDocument();
    expect(screen.getByTestId('line-b')).toBeInTheDocument();
    expect(screen.getByTestId('line-c')).toBeInTheDocument();
    expect(screen.queryByTestId('line-d')).not.toBeInTheDocument();
  });

  test('does not reserve horizontal space for runs without selected assessment values', () => {
    const runs = [
      ...Array.from({ length: 20 }, (_, i) => makeRun('empty-' + i, { startTime: i, metrics: [['other', 500]] })),
      makeRun('only-scored', { startTime: 30, metrics: [['correctness', 0]] }),
    ];
    renderChart({ runs });
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '1');
    expect(screen.getByTestId('line-correctness')).toHaveAttribute('data-values', '[0]');
  });

  test('places chronology above the legend and includes the run creation time in line tooltips', async () => {
    renderChart({
      runs: [
        makeRun('dated-run', {
          startTime: Date.UTC(2026, 8, 25, 12),
          metrics: [['correctness', 0.8]],
        }),
      ],
    });
    const chronologyLabels = screen.getByTestId('eval-runs-chronology-labels');
    expect(within(chronologyLabels).getByText('Earliest run')).toBeInTheDocument();
    expect(within(chronologyLabels).getByText('Latest run')).toBeInTheDocument();
    expect(screen.queryByText('Colors represent assessments')).not.toBeInTheDocument();
    expect(within(screen.getByTestId('tooltip')).getByText('Run')).toBeInTheDocument();
    expect(within(screen.getByTestId('tooltip')).getByTestId('eval-runs-tooltip-run-color')).toBeInTheDocument();
    expect(screen.getByTestId('tooltip')).toHaveTextContent('Created: Sep 25, 2026');
    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.queryByTestId('eval-runs-chronology-labels')).not.toBeInTheDocument();
    expect(screen.getByText('Colors represent runs')).toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: /Runs|Assessments/ })).not.toBeInTheDocument();
  });

  test('keeps dense charts within the panel and supports x-axis drag zoom with double-click reset', async () => {
    const metricKeys = ['correctness', 'relevance', 'safety'];
    const runs = Array.from({ length: 20 }, (_, index) =>
      makeRun(`run-${index}`, {
        metrics: metricKeys.map((metric, metricIndex) => [metric, (index + metricIndex) / 25]),
      }),
    );
    renderChart({ runs, metricKeys, selectedMetricKeys: metricKeys });
    const user = userEvent.setup();

    const chartArea = screen.getByTestId('eval-runs-chart-area');
    expect(chartArea).toHaveStyle({ width: '100%', minWidth: 0 });
    expect(screen.getByTestId('eval-runs-chart-viewport')).toHaveStyle({
      overflow: 'hidden',
    });

    const lineChart = screen.getByTestId('line-chart');
    fireEvent.pointerDown(lineChart);
    fireEvent.pointerMove(lineChart);
    fireEvent.pointerUp(lineChart);
    expect(lineChart).toHaveAttribute('data-count', '2');

    await user.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', String(metricKeys.length));

    await user.click(screen.getByRole('radio', { name: 'Line chart' }));
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '2');
    await user.dblClick(screen.getByTestId('line-chart'));
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', String(runs.length));

    await user.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', String(metricKeys.length));
    const barChart = screen.getByTestId('bar-chart');
    fireEvent.pointerDown(barChart);
    fireEvent.pointerMove(barChart);
    fireEvent.pointerUp(barChart);
    expect(barChart).toHaveAttribute('data-count', '2');
    expect(screen.getAllByTestId('cell')).toHaveLength(runs.length * 2);

    await user.dblClick(barChart);
    expect(barChart).toHaveAttribute('data-count', String(metricKeys.length));
  });

  test('ignores a one-run drag and offers a clear zoom button for a wider selection', async () => {
    renderChart({
      runs: Array.from({ length: 3 }, (_, index) =>
        makeRun(`run-${index}`, { startTime: index, metrics: [['score', index / 3]] }),
      ),
      metricKeys: ['score'],
      selectedMetricKeys: ['score'],
    });
    const lineChart = screen.getByTestId('line-chart');

    fireEvent.mouseDown(lineChart, { clientX: 0 });
    fireEvent.mouseMove(lineChart, { clientX: 25 });
    fireEvent.mouseUp(lineChart, { clientX: 25 });
    expect(lineChart).toHaveAttribute('data-count', '3');
    expect(screen.queryByText('Clear zoom')).not.toBeInTheDocument();

    fireEvent.mouseDown(lineChart, { clientX: 0 });
    fireEvent.mouseMove(lineChart, { clientX: 60 });
    fireEvent.mouseUp(lineChart, { clientX: 60 });
    expect(lineChart).toHaveAttribute('data-count', '2');
    await userEvent.click(screen.getByText('Clear zoom'));
    expect(lineChart).toHaveAttribute('data-count', '3');
  });

  test('zooms to an assessment group while preserving each run series', async () => {
    const onRunClick = jest.fn();
    const metricKeys = ['correctness', 'relevance', 'safety'];
    const runs = Array.from({ length: 2 }, (_, index) =>
      makeRun(`run-${index}`, {
        metrics: metricKeys.map((metric, metricIndex) => [metric, (index + metricIndex) / 5]),
      }),
    );
    renderChart({ runs, metricKeys, selectedMetricKeys: metricKeys, onRunClick });
    const user = userEvent.setup();

    await user.click(screen.getByRole('radio', { name: 'Bar chart' }));
    const barChart = screen.getByTestId('bar-chart');
    expect(barChart).toHaveAttribute('data-count', String(metricKeys.length));

    fireEvent.mouseDown(barChart, { clientX: 0 });
    fireEvent.mouseUp(barChart, { clientX: 0 });
    expect(barChart).toHaveAttribute('data-count', String(metricKeys.length));

    fireEvent.mouseDown(barChart, { clientX: 0 });
    fireEvent.mouseMove(barChart, { clientX: 60 });
    fireEvent.mouseUp(barChart, { clientX: 60 });
    fireEvent.click(screen.getByRole('img', { name: 'run-0, correctness: 0%' }));

    expect(barChart).toHaveAttribute('data-count', '2');
    expect(screen.getAllByTestId('cell')).toHaveLength(runs.length * 2);
    expect(onRunClick).not.toHaveBeenCalled();
  });

  test('defaults follow selected runs, including unbounded-only and empty runs', async () => {
    const runs = [
      makeRun('bounded', { metrics: [['quality', 0.8]] }),
      ...Array.from({ length: 4 }, (_, index) =>
        makeRun('unbounded-' + index, {
          metrics: [
            ['tokens', 10000 + index],
            ['latency', 15 + index],
            ['negative', -2],
          ],
        }),
      ),
      makeRun('empty', { metrics: [] }),
    ];
    // More than five unavailable columns must not crowd out the fallback.
    const props = {
      runs,
      metricKeys: [
        'quality',
        'missing1',
        'missing2',
        'missing3',
        'missing4',
        'missing5',
        'tokens',
        'latency',
        'negative',
      ],
      selectedMetricKeys: undefined,
      filterToSelected: true,
      onSelectedMetricKeysChange: jest.fn(),
    };
    const { rerender } = renderChart(props);
    expect(screen.getByTestId('line-quality')).toBeInTheDocument();
    rerender(
      chartTree({
        ...props,
        selectedRunUuids: ['unbounded-0', 'unbounded-1', 'unbounded-2', 'unbounded-3'],
      }),
    );
    expect(screen.queryByTestId('line-quality')).not.toBeInTheDocument();
    expect(screen.getByTestId('line-tokens')).toHaveAttribute('data-values', '[10000,10001,10002,10003]');
    expect(screen.getByTestId('line-latency')).toBeInTheDocument();
    expect(screen.getByTestId('line-negative')).toBeInTheDocument();
    expect(screen.queryByRole('switch', { name: 'Show all runs' })).not.toBeInTheDocument();
    rerender(chartTree({ ...props, selectedRunUuids: [] }));
    expect(screen.getByTestId('line-quality')).toBeInTheDocument();
    expect(screen.queryByTestId('line-tokens')).not.toBeInTheDocument();
    rerender(
      chartTree({
        ...props,
        selectedRunUuids: ['unbounded-0', 'unbounded-1', 'unbounded-2', 'unbounded-3'],
      }),
    );
    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.getByRole('img', { name: 'unbounded-0, tokens: 10000' })).toBeInTheDocument();
    rerender(chartTree({ ...props, selectedRunUuids: ['empty'] }));
    expect(screen.getByText('No assessment values to chart for the selected runs.')).toBeInTheDocument();
  });

  test('temporarily falls back for unavailable explicit choices and restores them when runs change', () => {
    const props = {
      runs: [
        makeRun('bounded', { metrics: [['quality', 0.8]] }),
        makeRun('unbounded', { metrics: [['tokens', 12345]] }),
      ],
      metricKeys: ['quality', 'tokens'],
      selectedMetricKeys: ['quality'],
      selectedRunUuids: ['unbounded'],
      filterToSelected: true,
      onSelectedMetricKeysChange: jest.fn(),
    };
    const { rerender } = renderChart(props);
    expect(screen.getByTestId('line-tokens')).toBeInTheDocument();
    expect(props.onSelectedMetricKeysChange).not.toHaveBeenCalled();
    rerender(chartTree({ ...props, selectedRunUuids: ['bounded'] }));
    expect(screen.getByTestId('line-quality')).toBeInTheDocument();
    rerender(chartTree({ ...props, selectedMetricKeys: [] }));
    expect(screen.queryByTestId('line-tokens')).not.toBeInTheDocument();
  });

  test('keeps the bar tooltip inside the viewport and beside tall bars', () => {
    expect(
      getEvalRunsBarTooltipPosition({ left: -20, top: 2, bottom: 42, width: 20 }, { width: 100, height: 30 }, 200, 100),
    ).toEqual({ left: 8, top: 8 });

    expect(
      getEvalRunsBarTooltipPosition(
        { left: 190, top: 80, bottom: 90, width: 20 },
        { width: 100, height: 30 },
        200,
        100,
      ),
    ).toEqual({ left: 92, top: 42 });

    expect(
      getEvalRunsBarTooltipPosition(
        { left: 150, top: 10, bottom: 190, width: 20 },
        { width: 100, height: 60 },
        200,
        220,
      ),
    ).toEqual({ left: 42, top: 10 });
  });

  test('defaults to a line chart with all loaded runs in chronological order', async () => {
    renderChart({
      runs: Array.from({ length: 6 }, (_, index) =>
        makeRun(`run-${index}`, {
          startTime: index + 1,
          metrics: [['correctness', 0.5 + index / 10]],
        }),
      ).reverse(),
    });

    expect(screen.queryByText('Evaluation runs')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Toggle evaluation runs chart' })).not.toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Line chart' })).toBeChecked();
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '6');
    expect(screen.queryByText('Recent 5 runs')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.getAllByRole('img').map((bar) => bar.getAttribute('aria-label'))).toEqual([
      'run-0, correctness: 50%',
      'run-1, correctness: 60%',
      'run-2, correctness: 70%',
      'run-3, correctness: 80%',
      'run-4, correctness: 90%',
      'run-5, correctness: 100%',
    ]);
  });

  test('connects missing assessment values without fabricating scores', () => {
    renderChart({
      runs: [
        makeRun('run-a', { startTime: 1, metrics: [['quality.score', 0.8]] }),
        makeRun('run-b', { startTime: 2, metrics: [] }),
        makeRun('run-c', { startTime: 3, metrics: [['quality.score', 0.6]] }),
      ],
      metricKeys: ['quality.score'],
      selectedMetricKeys: ['quality.score'],
    });

    const line = screen.getByTestId('line-quality.score');
    expect(line).toHaveAttribute('data-key', 'scores.m0');
    expect(line).toHaveAttribute('data-values', JSON.stringify([0.8, 0.6]));
    expect(line).toHaveAttribute('data-connect-nulls', 'true');
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '2');
  });

  test('does not render line dots for missing values while preserving zero values', () => {
    const lineSpy = jest.spyOn(rechartsSpies, 'Line');
    const onRunClick = jest.fn();
    try {
      renderChart({
        runs: [
          makeRun('missing', { startTime: 1, metrics: [] }),
          makeRun('zero', { startTime: 2, metrics: [['score', 0]] }),
        ],
        metricKeys: ['score'],
        selectedMetricKeys: ['score'],
        onRunClick,
      });

      type LineDotProps = {
        cx?: number;
        cy?: number;
        payload?: unknown;
        value?: number | null;
      };
      type LineProps = {
        activeDot?: boolean;
        dot?: (props: LineDotProps) => unknown;
        name?: string;
      };
      const lineProps = lineSpy.mock.calls.map(([props]) => props as LineProps).find(({ name }) => name === 'score');
      expect(lineProps?.dot).toEqual(expect.any(Function));
      expect(lineProps?.activeDot).toBe(false);

      const renderDot = lineProps!.dot!;
      expect(renderDot({ value: null, cx: 10, cy: 20, payload: { runUuid: 'missing' } })).toBeNull();
      expect(renderDot({ value: undefined, cx: 10, cy: 20, payload: { runUuid: 'missing' } })).toBeNull();
      expect(renderDot({ value: 0, cx: 10, cy: 20, payload: { runUuid: 'zero' } })).toEqual(
        expect.objectContaining({
          props: expect.objectContaining({
            children: expect.arrayContaining([
              expect.objectContaining({ props: expect.objectContaining({ cx: 10, cy: 20, r: 12 }) }),
            ]),
          }),
        }),
      );
      const dot = renderDot({ value: 0, cx: 10, cy: 20, payload: { runUuid: 'zero' } }) as {
        props: { onClick: (event: { stopPropagation: () => void }) => void };
      };
      const lineChart = screen.getByTestId('line-chart');
      fireEvent.mouseDown(lineChart, { clientX: 0 });
      fireEvent.mouseMove(lineChart, { clientX: 25 });
      fireEvent.mouseUp(lineChart, { clientX: 25 });
      dot.props.onClick({ stopPropagation: jest.fn() });
      expect(onRunClick).toHaveBeenCalledWith('zero');
    } finally {
      jest.restoreAllMocks();
    }
  });

  test('leaves headroom above non-fractional maxima and preserves mixed-value bounds', () => {
    expect(getEvalRunsYAxisDomain([0, 25])[0]).toBeLessThanOrEqual(0);
    expect(getEvalRunsYAxisDomain([0, 25])[1]).toBeGreaterThan(25);
    expect(getEvalRunsYAxisDomain([-1, 25])[0]).toBeLessThanOrEqual(-1);
    expect(getEvalRunsYAxisDomain([-1, 25])[1]).toBeGreaterThan(25);
    expect(getEvalRunsYAxisDomain([0, 0])[0]).toBeLessThan(0);
    expect(getEvalRunsYAxisDomain([0, 0])[1]).toBeGreaterThan(0);
  });

  test('uses the expanded domain and pixel padding for a series at the data maximum', async () => {
    const yAxisSpy = jest.spyOn(rechartsSpies, 'YAxis');
    try {
      renderChart({
        runs: [
          makeRun('low', {
            metrics: [
              ['eval_cases', 0],
              ['valid_response/mean', -1],
            ],
          }),
          makeRun('high', {
            metrics: [
              ['eval_cases', 25],
              ['valid_response/mean', 0],
            ],
          }),
        ],
        metricKeys: ['eval_cases', 'valid_response/mean'],
        selectedMetricKeys: ['eval_cases', 'valid_response/mean'],
      });

      type YAxisProps = {
        domain?: unknown;
        padding?: { top?: number; bottom?: number };
      };
      const getLatestYAxisProps = () => yAxisSpy.mock.calls.at(-1)?.[0] as YAxisProps;
      const lineYAxisProps = getLatestYAxisProps();
      expect(typeof lineYAxisProps.domain).toBe('function');
      const lineDomain = (lineYAxisProps.domain as (domain: [number, number]) => [number, number])([0, 25]);
      expect(lineDomain[1]).toBeGreaterThan(25);
      expect(lineYAxisProps.padding?.top).toBeGreaterThan(0);
      expect(lineYAxisProps.padding?.bottom).toBeGreaterThan(0);
      expect(screen.getByTestId('line-eval_cases')).toHaveAttribute('data-values', '[0,25]');

      await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
      const barYAxisProps = getLatestYAxisProps();
      expect(typeof barYAxisProps.domain).toBe('function');
      const barDomain = (barYAxisProps.domain as (domain: [number, number]) => [number, number])([1000, 5000]);
      expect(barDomain[0]).toBe(0);
      expect(barDomain[1]).toBeGreaterThan(5000);
      expect(barYAxisProps.padding?.top).toBeGreaterThan(0);
      expect(barYAxisProps.padding?.bottom).toBeGreaterThan(0);
    } finally {
      jest.restoreAllMocks();
    }
  });

  test('keeps a run with any selected score while preserving gaps in its other scores', () => {
    renderChart({
      runs: [
        makeRun('first', { startTime: 1, metrics: [['quality', 0.8]] }),
        makeRun('middle', { startTime: 2, metrics: [['accuracy', 0.5]] }),
        makeRun('last', { startTime: 3, metrics: [['quality', 0.6]] }),
      ],
      metricKeys: ['quality', 'accuracy'],
      selectedMetricKeys: ['quality', 'accuracy'],
    });
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    expect(screen.getByTestId('line-quality')).toHaveAttribute('data-values', '[0.8,null,0.6]');
    expect(screen.getByTestId('line-accuracy')).toHaveAttribute('data-values', '[null,0.5,null]');
  });

  test('limits displayed fractional chart values to four decimal places after percent conversion', async () => {
    renderChart({
      runs: [makeRun('run-a', { metrics: [['correctness', 0.123456789]] })],
      metricKeys: ['correctness'],
      selectedMetricKeys: ['correctness'],
    });

    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));

    expect(screen.getByRole('img', { name: 'run-a, correctness: 12.3457%' })).toBeInTheDocument();
  });

  test('limits displayed non-fractional chart values to four decimal places', async () => {
    renderChart({
      runs: [makeRun('run-a', { metrics: [['latency', 123.456789]] })],
      metricKeys: ['latency'],
      selectedMetricKeys: ['latency'],
    });

    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));

    expect(screen.getByRole('img', { name: 'run-a, latency: 123.4568' })).toBeInTheDocument();
  });

  test('charts the controlled assessment selection and keeps table order when selecting one', async () => {
    const onSelectedMetricKeysChange = jest.fn();
    const { rerender } = renderChart({
      runs: [
        makeRun('run-1', {
          metrics: [
            ['first.score', 0.8],
            ['second/score', 0.7],
            ['third score', 0.6],
          ],
        }),
      ],
      metricKeys: ['first.score', 'second/score', 'third score'],
      selectedMetricKeys: ['first.score', 'second/score'],
      onSelectedMetricKeysChange,
    });

    await userEvent.click(screen.getByRole('button', { name: /2 assessments/ }));
    expect(screen.getByRole('menuitemcheckbox', { name: 'first.score' })).toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'second/score' })).toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'third score' })).not.toBeChecked();

    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'third score' }));
    expect(onSelectedMetricKeysChange).toHaveBeenCalledWith(['first.score', 'second/score', 'third score']);

    rerender(
      chartTree({
        runs: [
          makeRun('run-1', {
            metrics: [
              ['first.score', 0.8],
              ['second/score', 0.7],
              ['third score', 0.6],
            ],
          }),
        ],
        metricKeys: ['first.score', 'second/score', 'third score'],
        selectedMetricKeys: ['third score'],
        onSelectedMetricKeysChange,
      }),
    );
    expect(screen.getByRole('button', { name: 'third score' })).toBeInTheDocument();
  });

  test('preserves line colors when the first assessment is unchecked and rechecked', async () => {
    const metricKeys = ['quality', 'accuracy', 'relevance'];
    const props = {
      runs: [
        makeRun('run-a', {
          metrics: [
            ['quality', 0.8],
            ['accuracy', 0.7],
            ['relevance', 0.6],
          ],
        }),
      ],
      metricKeys,
      selectedMetricKeys: metricKeys,
      onSelectedMetricKeysChange: jest.fn(),
    };
    const { rerender } = renderChart(props);
    const initialColors = metricKeys.map((metric) => screen.getByTestId(`line-${metric}`).getAttribute('data-stroke'));
    initialColors.forEach((color) => expect(color).toBeTruthy());
    expect(new Set(initialColors).size).toBe(metricKeys.length);

    await userEvent.click(screen.getByRole('button', { name: 'All assessments' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'quality' }));
    expect(props.onSelectedMetricKeysChange).toHaveBeenLastCalledWith(['accuracy', 'relevance']);
    rerender(chartTree({ ...props, selectedMetricKeys: ['accuracy', 'relevance'] }));

    expect(screen.getByRole('menuitemcheckbox', { name: 'quality' })).not.toBeChecked();
    expect(screen.queryByTestId('line-quality')).not.toBeInTheDocument();
    expect(screen.getByTestId('line-accuracy')).toHaveAttribute('data-stroke', initialColors[1]);
    expect(screen.getByTestId('line-relevance')).toHaveAttribute('data-stroke', initialColors[2]);

    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'quality' }));
    expect(props.onSelectedMetricKeysChange).toHaveBeenLastCalledWith(['accuracy', 'relevance', 'quality']);
    rerender(
      chartTree({
        ...props,
        selectedMetricKeys: ['accuracy', 'relevance', 'quality'],
      }),
    );

    expect(screen.getByRole('menuitemcheckbox', { name: 'quality' })).toBeChecked();
    metricKeys.forEach((metric, index) => {
      expect(screen.getByTestId(`line-${metric}`)).toHaveAttribute('data-stroke', initialColors[index]);
    });
  });

  test('keeps multiple focused assessments while temporarily highlighting a hovered legend entry', () => {
    const legendSpy = jest.spyOn(rechartsSpies, 'Legend');
    const lineSpy = jest.spyOn(rechartsSpies, 'Line');
    try {
      renderChart({
        runs: [
          makeRun('run-a', {
            metrics: [
              ['A', 0.8],
              ['B', 0.7],
              ['C', 0.6],
            ],
          }),
        ],
        metricKeys: ['A', 'B', 'C'],
        selectedMetricKeys: ['A', 'B', 'C'],
      });

      type LegendProps = {
        onClick?: (entry: unknown) => void;
        onMouseEnter?: (entry: unknown) => void;
        onMouseLeave?: () => void;
      };
      type LineProps = { name?: string; strokeOpacity?: number };
      const getLegendProps = () => legendSpy.mock.calls.at(-1)?.[0] as LegendProps;
      const getLineOpacity = (metric: string) =>
        lineSpy.mock.calls
          .map(([props]) => props as LineProps)
          .filter(({ name }) => name === metric)
          .at(-1)?.strokeOpacity;

      const expectOpacities = (expected: Record<string, number>) => {
        Object.entries(expected).forEach(([metric, opacity]) => expect(getLineOpacity(metric)).toBe(opacity));
      };

      expectOpacities({ A: 1, B: 1, C: 1 });
      act(() => getLegendProps().onClick?.({ value: 'A' }));
      expectOpacities({ A: 1, B: 0.55, C: 0.55 });

      act(() => getLegendProps().onClick?.({ value: 'B' }));
      expectOpacities({ A: 1, B: 1, C: 0.55 });

      act(() => getLegendProps().onMouseEnter?.({ value: 'C' }));
      expectOpacities({ A: 1, B: 1, C: 1 });

      act(() => getLegendProps().onMouseLeave?.());
      expectOpacities({ A: 1, B: 1, C: 0.55 });
    } finally {
      jest.restoreAllMocks();
    }
  });

  test('preserves hidden selections when editing visible assessments under a filter', async () => {
    const onChange = jest.fn();
    const props = {
      runs: [
        makeRun('a', {
          metrics: [
            ['A', 0.8],
            ['B', 0.7],
          ],
        }),
      ],
      metricKeys: ['A'],
      selectedMetricKeys: ['A', 'B'],
      onSelectedMetricKeysChange: onChange,
    };
    const { rerender } = renderChart(props);
    await userEvent.click(screen.getByRole('button', { name: 'All assessments' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'A' }));
    expect(onChange).toHaveBeenLastCalledWith(['B']);
    rerender(
      chartTree({
        ...props,
        metricKeys: ['A', 'B'],
        selectedMetricKeys: ['B'],
      }),
    );
    expect(screen.getByTestId('line-B')).toBeInTheDocument();
    expect(screen.queryByTestId('line-A')).not.toBeInTheDocument();

    rerender(chartTree(props));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'All assessments' }));
    expect(onChange).toHaveBeenLastCalledWith(['B']);
  });

  test('offers assessment-axis bars with table run colors and correct hover values', async () => {
    renderChart({
      runs: [
        makeRun('run-a', {
          metrics: [
            ['quality', 0.8],
            ['accuracy', 0.6],
          ],
        }),
        makeRun('run-b', {
          metrics: [
            ['quality', 0.7],
            ['accuracy', 0.9],
          ],
        }),
      ],
      metricKeys: ['quality', ...Array.from({ length: 8 }, (_, index) => `unused-${index}`), 'accuracy'],
      selectedMetricKeys: ['quality', 'accuracy'],
    });
    expect(screen.getByTestId('x-axis')).toHaveAttribute('data-hide', 'true');
    expect(screen.getByTestId('line-quality').getAttribute('data-stroke')).not.toEqual(
      screen.getByTestId('line-accuracy').getAttribute('data-stroke'),
    );
    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.queryByRole('radio', { name: /Runs|Assessments/ })).not.toBeInTheDocument();
    expect(screen.getByTestId('bar-run-a')).toHaveAttribute('data-fill', '#123456');
    expect(screen.getByTestId('bar-run-b')).toHaveAttribute('data-fill', '#654321');
    expect(screen.getByTestId('x-axis')).toHaveAttribute('data-hide', 'false');
    expect(screen.getByTestId('x-axis')).toHaveAttribute('data-key', 'axisKey');
    fireEvent.mouseEnter(screen.getByRole('img', { name: 'run-b, accuracy: 90%' }));
    const tooltip = screen.getByTestId('eval-runs-bar-tooltip');
    expect(within(tooltip).getByText('Run')).toBeInTheDocument();
    expect(within(tooltip).getByTestId('eval-runs-tooltip-run-color')).toHaveStyle({ backgroundColor: '#654321' });
    expect(tooltip).toHaveTextContent('run-b');
    expect(screen.getByTestId('eval-runs-bar-tooltip')).toHaveTextContent('90%');
    expect(screen.getByRole('img', { name: 'run-b, quality: 70%' })).toHaveAttribute('data-fill-opacity', '1');
    expect(screen.getByRole('img', { name: 'run-a, quality: 80%' })).toHaveAttribute('data-fill-opacity', '0.55');
    expect(screen.queryByRole('radio', { name: 'Runs' })).not.toBeInTheDocument();
  });

  test('All assessments selects every assessment column', async () => {
    const onSelectedMetricKeysChange = jest.fn();
    renderChart({
      metricKeys: ['first', 'second', 'third'],
      selectedMetricKeys: ['first'],
      onSelectedMetricKeysChange,
    });

    await userEvent.click(screen.getByRole('button', { name: 'first' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'All assessments' }));

    expect(onSelectedMetricKeysChange).toHaveBeenCalledWith(['first', 'second', 'third']);
  });

  test('All assessments clears every assessment when all are already selected', async () => {
    const onSelectedMetricKeysChange = jest.fn();
    renderChart({
      metricKeys: ['first', 'second', 'third'],
      selectedMetricKeys: ['first', 'second', 'third'],
      onSelectedMetricKeysChange,
    });

    await userEvent.click(screen.getByRole('button', { name: 'All assessments' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'All assessments' }));

    expect(onSelectedMetricKeysChange).toHaveBeenCalledWith([]);
  });

  test('leaves dataset selection to the table toolbar', async () => {
    renderChart({
      runs: [
        makeRun('dataset-a-run', {
          startTime: 1,
          datasetName: 'dataset-a',
          metrics: [['score', 0.8]],
        }),
        makeRun('dataset-b-run', {
          startTime: 2,
          datasetName: 'dataset-b',
          metrics: [['score', 0.7]],
        }),
      ],
      metricKeys: ['score'],
      selectedMetricKeys: ['score'],
    });

    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '2');
    expect(
      screen.queryByRole('button', {
        name: /All datasets|Filtered datasets|dataset-b/i,
      }),
    ).not.toBeInTheDocument();
  });

  test('keeps selected runs with recent runs, highlights them, and filters only when requested', async () => {
    const onRunClick = jest.fn();
    const props = {
      runs: [
        makeRun('run-1', { startTime: 1, metrics: [['score', 0.5]] }),
        makeRun('run-2', { startTime: 2, metrics: [['score', 0.6]] }),
        makeRun('run-3', { startTime: 3, metrics: [['score', 0.7]] }),
      ],
      selectedRunUuids: ['run-2'],
      metricKeys: ['score'],
      selectedMetricKeys: ['score'],
      onRunClick,
    };
    const { rerender } = renderChart(props);

    expect(screen.queryByText('Evaluation runs')).not.toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    expect(screen.queryByRole('switch', { name: 'Show all runs' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    expect(screen.getByRole('img', { name: 'run-2, score: 60%' })).toHaveAttribute('data-fill-opacity', '1');
    expect(screen.getByRole('img', { name: 'run-1, score: 50%' })).toHaveAttribute('data-fill-opacity', '0.55');

    fireEvent.mouseEnter(screen.getByRole('img', { name: 'run-1, score: 50%' }));
    fireEvent.click(screen.getByRole('img', { name: 'run-1, score: 50%' }));
    expect(onRunClick).toHaveBeenCalledWith('run-1');

    rerender(
      chartTree({
        ...props,
        filterToSelected: true,
        onSelectedMetricKeysChange: jest.fn(),
      }),
    );
    expect(screen.queryByText('1 selected run')).not.toBeInTheDocument();
    expect(screen.getByTestId('bar-chart')).toHaveAttribute('data-count', '1');
    expect(screen.getByRole('img', { name: 'run-2, score: 60%' })).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: 'run-1, score: 50%' })).not.toBeInTheDocument();
  });

  test('falls back to all loaded runs when checked ids are hidden by the shared filter', () => {
    renderChart({
      runs: [makeRun('visible', { metrics: [['score', 0.5]] })],
      selectedRunUuids: ['hidden'],
      metricKeys: ['score'],
      selectedMetricKeys: ['score'],
    });

    expect(screen.queryByText('Evaluation runs')).not.toBeInTheDocument();
    expect(screen.queryByRole('switch', { name: 'Show all runs' })).not.toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '1');
  });

  test('renders bars after switching modes and keeps punctuation-safe assessment keys', async () => {
    renderChart({
      runs: [makeRun('run-a', { metrics: [['quality.score/v1', 0.9]] })],
      metricKeys: ['quality.score/v1'],
      selectedMetricKeys: ['quality.score/v1'],
    });

    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    const bar = await screen.findByRole('img', {
      name: 'run-a, quality.score/v1: 90%',
    });
    expect(bar).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Bar chart' })).toBeChecked();
  });

  test('shows a loading skeleton and removes the operational duration picker', async () => {
    renderChart({
      runs: [],
      isLoading: true,
      metricKeys: ['score'],
      selectedMetricKeys: ['score'],
    });

    expect(screen.queryByText('Evaluation runs')).not.toBeInTheDocument();
    expect(screen.getByText('Loading evaluation runs chart')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'All assessments' }));
    expect(screen.queryByText('Duration (min)')).not.toBeInTheDocument();
    expect(screen.queryByText('Operational')).not.toBeInTheDocument();
  });

  test('shows the bar tooltip when the pointer is near a narrow bar', async () => {
    renderChart({
      runs: [
        makeRun('run-a', { startTime: 1, metrics: [['correctness', 0.9]] }),
        makeRun('run-b', { startTime: 2, metrics: [['correctness', 0.8]] }),
      ],
      metricKeys: ['correctness'],
      selectedMetricKeys: ['correctness'],
    });
    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    const barA = await screen.findByRole('img', { name: 'run-a, correctness: 90%' });
    const barB = screen.getByRole('img', { name: 'run-b, correctness: 80%' });
    jest.spyOn(barA, 'getBoundingClientRect').mockReturnValue(makeBoundingRect(20, 40, 4, 60));
    jest.spyOn(barB, 'getBoundingClientRect').mockReturnValue(makeBoundingRect(40, 50, 4, 50));
    const chartArea = screen.getByTestId('eval-runs-chart-area');

    fireEvent.mouseMove(chartArea, { clientX: 47, clientY: 70 });
    expect(screen.getByTestId('eval-runs-bar-tooltip')).toHaveTextContent('run-b');

    fireEvent.mouseMove(chartArea, { clientX: 26, clientY: 30 });
    expect(screen.getByTestId('eval-runs-bar-tooltip')).toHaveTextContent('run-a');

    fireEvent.mouseMove(chartArea, { clientX: 22, clientY: 120 });
    expect(screen.queryByTestId('eval-runs-bar-tooltip')).not.toBeInTheDocument();

    fireEvent.mouseMove(chartArea, { clientX: 80, clientY: 70 });
    expect(screen.queryByTestId('eval-runs-bar-tooltip')).not.toBeInTheDocument();
  });

  test('dismisses the bar tooltip after scrolling and resizing', async () => {
    renderChart({
      runs: [
        makeRun('run-a', { startTime: 1, metrics: [['correctness', 0.9]] }),
        makeRun('run-b', { startTime: 2, metrics: [['correctness', 0.8]] }),
      ],
      metricKeys: ['correctness'],
      selectedMetricKeys: ['correctness'],
    });
    await userEvent.click(screen.getByRole('radio', { name: 'Bar chart' }));
    const bar = await screen.findByRole('img', {
      name: 'run-b, correctness: 80%',
    });
    jest.spyOn(bar, 'getBoundingClientRect').mockReturnValue(makeBoundingRect(20, 40, 20, 20));

    fireEvent.mouseEnter(bar);
    expect(screen.getByTestId('eval-runs-bar-tooltip')).toBeInTheDocument();
    expect(screen.getByTestId('eval-runs-bar-tooltip')).toHaveTextContent('run-b');

    fireEvent.scroll(window);
    expect(screen.queryByTestId('eval-runs-bar-tooltip')).not.toBeInTheDocument();

    fireEvent.mouseLeave(bar);
    fireEvent.mouseEnter(bar);
    expect(screen.getByTestId('eval-runs-bar-tooltip')).toBeInTheDocument();
    fireEvent.resize(window);
    expect(screen.queryByTestId('eval-runs-bar-tooltip')).not.toBeInTheDocument();
  });
});

describe('default chart assessments', () => {
  test('prefers bounded means including zero and one without filling remaining slots', () => {
    const runs = [
      makeRun('a', {
        metrics: [
          ['tokens', 1000],
          ['zero', 0],
          ['one', 1],
          ['mean', 0],
          ['bad', NaN],
        ],
      }),
      makeRun('b', {
        metrics: [
          ['tokens', 2000],
          ['zero', 0],
          ['one', 1],
          ['mean', 2],
        ],
      }),
    ];
    expect(getDefaultChartMetricKeys(runs, ['tokens', 'one', 'mean', 'zero', 'bad'])).toEqual(['one', 'mean', 'zero']);
  });

  test('falls back to the first five metrics when none have bounded means', () => {
    const keys = ['a', 'b', 'c', 'd', 'e', 'f'];
    const runs = [makeRun('a', { metrics: keys.map((key) => [key, 100]) })];
    expect(getDefaultChartMetricKeys(runs, keys)).toEqual(keys.slice(0, 5));
    expect(getDefaultChartMetricKeys([], [])).toEqual([]);
  });

  test('supports a smaller assessment limit for compact charts', () => {
    const keys = ['a', 'b', 'c', 'd'];
    const runs = [makeRun('a', { metrics: keys.map((key) => [key, 0.5]) })];
    expect(getDefaultChartMetricKeys(runs, keys, 3)).toEqual(keys.slice(0, 3));
  });
});

describe('formatEvalRunsXAxisTick', () => {
  test('adapts labels to the available category width', () => {
    expect(formatEvalRunsXAxisTick('long assessment name', 200, 2)).toBe('long ass…');
    expect(formatEvalRunsXAxisTick('long assessment name', 100, 2)).toBe('l…');
  });

  test('keeps labels intact until a chart width is available', () => {
    expect(formatEvalRunsXAxisTick('long assessment name', 0, 2)).toBe('long assessment name');
  });
});
