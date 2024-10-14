import { render, screen, waitFor, within } from '@testing-library/react';
import { RunsChartType, RunsChartsLineCardConfig } from '../runs-charts.types';
import { RunsChartsLineChartXAxisType } from './RunsCharts.common';
import { RunsChartsGlobalChartSettingsDropdown } from './RunsChartsGlobalChartSettingsDropdown';
import { useState } from 'react';
import { createExperimentPageUIState, ExperimentPageUIState } from '../../experiment-page/models/ExperimentPageUIState';
import { compact, noop } from 'lodash';
import { RunsChartsCard } from './cards/RunsChartsCard';
import { RunsChartsTooltipWrapper } from '../hooks/useRunsChartsTooltip';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { DragAndDropProvider } from '../../../../common/hooks/useDragAndDropElement';
import { IntlProvider } from 'react-intl';
import { RunsMetricsLinePlot } from './RunsMetricsLinePlot';
import userEvent from '@testing-library/user-event-14';
import { DesignSystemProvider } from '@databricks/design-system';
import {
  ExperimentPageUIStateContextProvider,
  useUpdateExperimentViewUIState,
} from '../../experiment-page/contexts/ExperimentPageUIStateContext';

jest.setTimeout(30000); // Larger timeout for integration testing

// Enable feature shouldEnableGlobalLineChartConfig
jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../../../common/utils/FeatureUtils'),
  shouldEnableGlobalLineChartConfig: jest.fn().mockReturnValue(true),
}));

jest.mock('../hooks/useIsInViewport', () => ({
  useIsInViewport: jest
    .fn()
    .mockImplementation(() => ({ isInViewport: true, isInViewportDeferred: true, elementRef: undefined })),
}));

// mock line plot
jest.mock('./RunsMetricsLinePlot', () => ({
  RunsMetricsLinePlot: jest.fn(),
}));

describe('RunsChartsGlobalChartSettingsDropdown', () => {
  const commonChartProps = {
    isGenerated: true,
    deleted: false,
    scaleType: 'linear' as const,
    runsCountToCompare: 10,
    metricSectionId: 'metric-section-1',
    metricKey: '',
    selectedXAxisMetricKey: '',
    useGlobalLineSmoothing: false,
    useGlobalXaxisKey: false,
    range: {
      xMin: undefined,
      xMax: undefined,
      yMin: undefined,
      yMax: undefined,
    },
  };
  // We will have two test charts:
  // - first one will have lineSmoothness set to 50 and xAxisKey set to STEP
  // - second one will have lineSmoothness set to 30 and xAxisKey set to TIME
  const testCharts: RunsChartsLineCardConfig[] = [
    {
      type: RunsChartType.LINE,
      uuid: 'chart-alpha',
      lineSmoothness: 0,
      xAxisKey: RunsChartsLineChartXAxisType.STEP,
      xAxisScaleType: 'linear',
      selectedMetricKeys: ['alpha'],
      ...commonChartProps,
    },
    {
      type: RunsChartType.LINE,
      uuid: 'chart-beta',
      lineSmoothness: 30,
      xAxisKey: RunsChartsLineChartXAxisType.STEP,
      xAxisScaleType: 'linear',
      selectedMetricKeys: ['beta'],
      ...commonChartProps,
    },
  ];

  const renderTestComponent = () => {
    const TestComponent = () => {
      const [uiState, setUIState] = useState<ExperimentPageUIState>({
        ...createExperimentPageUIState(),
        compareRunCharts: testCharts,
      });

      return (
        <div>
          <ExperimentPageUIStateContextProvider setUIState={setUIState}>
            <RunsChartsGlobalChartSettingsDropdown
              globalLineChartConfig={uiState.globalLineChartConfig}
              updateUIState={(setter) => setUIState((current) => ({ ...current, ...setter(current) }))}
              metricKeyList={compact(testCharts.flatMap((chart) => chart.selectedMetricKeys))}
            />
            <div>
              <RunsChartsTooltipWrapper component={() => null} contextData={{}}>
                <DragAndDropProvider>
                  {uiState.compareRunCharts?.map((chartConfig, index) => (
                    <RunsChartsCard
                      key={chartConfig.uuid}
                      cardConfig={chartConfig}
                      chartRunData={[]}
                      index={index}
                      onRemoveChart={noop}
                      groupBy={null}
                      onReorderWith={noop}
                      onStartEditChart={noop}
                      sectionIndex={0}
                      canMoveDown
                      canMoveUp
                      globalLineChartConfig={uiState.globalLineChartConfig}
                    />
                  ))}
                </DragAndDropProvider>
              </RunsChartsTooltipWrapper>
            </div>
          </ExperimentPageUIStateContextProvider>
        </div>
      );
    };
    render(<TestComponent />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <MockedReduxStoreProvider
              state={{
                entities: { sampledMetricsByRunUuid: {} },
              }}
            >
              {children}
            </MockedReduxStoreProvider>
          </DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  beforeAll(() => {
    jest.mocked(RunsMetricsLinePlot).mockImplementation(({ selectedMetricKeys, lineSmoothness, xAxisKey }) => {
      const updateUIState = useUpdateExperimentViewUIState();
      const setUseGlobalSettings = (value: boolean) =>
        updateUIState((current) => {
          return {
            ...current,
            compareRunCharts: current.compareRunCharts?.map((chart) => {
              if (chart.uuid === `chart-${selectedMetricKeys?.join(',')}`) {
                return {
                  ...chart,
                  useGlobalLineSmoothing: value,
                  useGlobalXaxisKey: value,
                };
              }
              return chart;
            }),
          };
        });
      return (
        <div data-testid={`chart-${selectedMetricKeys?.join(',')}`}>
          x-axis: {xAxisKey}, smoothness: {lineSmoothness}
          <button onClick={() => setUseGlobalSettings(false)}>use local settings</button>
          <button onClick={() => setUseGlobalSettings(true)}>use global settings</button>
        </div>
      );
    });
  });

  test('it should apply proper settings to the chart', async () => {
    renderTestComponent();

    // Wait for the charts to render with their own settings
    await waitFor(() => {
      expect(screen.getByTestId('chart-alpha').textContent).toContain('x-axis: step');
      expect(screen.getByTestId('chart-beta').textContent).toContain('x-axis: step');
      expect(screen.getByTestId('chart-alpha').textContent).toContain('smoothness: 0');
      expect(screen.getByTestId('chart-beta').textContent).toContain('smoothness: 30');
    });

    // Open the dropdown
    await userEvent.click(screen.getByLabelText('Configure charts'));

    // Change the setting for x-axis to time
    await userEvent.click(screen.getByText('Time (wall)'));

    // Change "beta" chart configuration to use global settings
    await userEvent.click(within(screen.getByTestId('chart-beta')).getByText('use global settings'));

    // Expect "beta" metric chart to reflect the changes while "alpha" should stay the same
    expect(screen.getByTestId('chart-alpha').textContent).toContain('x-axis: step');
    expect(screen.getByTestId('chart-beta').textContent).toContain('x-axis: time');

    // Change the global x-axis type to "step"
    await userEvent.click(screen.getByLabelText('Configure charts'));
    await userEvent.click(screen.getByText('Step'));

    // Both charts should now have x-axis set to "step"
    expect(screen.getByTestId('chart-alpha').textContent).toContain('x-axis: step');
    expect(screen.getByTestId('chart-beta').textContent).toContain('x-axis: step');

    // Change the line smoothness to 42
    await userEvent.click(screen.getByLabelText('Configure charts'));
    await userEvent.clear(screen.getByRole('spinbutton'));
    await userEvent.type(screen.getByRole('spinbutton'), '42');

    // Expect beta chart to reflect the changes while alpha should stay the same
    expect(screen.getByTestId('chart-alpha').textContent).toContain('smoothness: 0');
    expect(screen.getByTestId('chart-beta').textContent).toContain('smoothness: 42');

    // Now, alpha chart will use global settings while beta will use local settings
    await userEvent.click(within(screen.getByTestId('chart-alpha')).getByText('use global settings'));
    await userEvent.click(within(screen.getByTestId('chart-beta')).getByText('use local settings'));

    // Expect mocked plots to reflect the changes
    expect(screen.getByTestId('chart-alpha').textContent).toContain('smoothness: 42');
    expect(screen.getByTestId('chart-beta').textContent).toContain('smoothness: 30');

    // Revert alpha chart to use local settings
    await userEvent.click(within(screen.getByTestId('chart-alpha')).getByText('use local settings'));

    // We should go back to original per-chart settings
    expect(screen.getByTestId('chart-alpha').textContent).toContain('x-axis: step');
    expect(screen.getByTestId('chart-beta').textContent).toContain('x-axis: step');
    expect(screen.getByTestId('chart-alpha').textContent).toContain('smoothness: 0');
    expect(screen.getByTestId('chart-beta').textContent).toContain('smoothness: 30');
  });
});
