import { render, screen, waitFor } from '@testing-library/react';
import { RunsChartsConfigureModal } from './RunsChartsConfigureModal';
import type { RunsChartsLineCardConfig } from '../runs-charts.types';
import { RunsChartType } from '../runs-charts.types';
import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { RunsMetricsLinePlot } from './RunsMetricsLinePlot';
import { last } from 'lodash';
import userEvent from '@testing-library/user-event';
import { RunsChartsLineChartXAxisType } from './RunsCharts.common';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';

// Larger timeout for integration testing (form rendering)
// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(15000);

// Mock <RunsMetricsLinePlot> component, it's exact implementation is not important for this test
jest.mock('./RunsMetricsLinePlot', () => ({
  RunsMetricsLinePlot: jest.fn(() => <div>RunsMetricsLinePlot</div>),
}));

const sampleChartData = [
  {
    displayName: 'test-run',
    images: {},
    metrics: {},
    params: {},
    tags: {},
    uuid: 'test-run-uuid',
  },
];

const sampleLineChartConfig: RunsChartsLineCardConfig = {
  uuid: 'some-uuid',
  type: RunsChartType.LINE,
  deleted: false,
  isGenerated: false,
  selectedMetricKeys: ['metric-a'],
  lineSmoothness: 0,
  metricKey: 'metric-a',
  scaleType: 'linear',
  xAxisScaleType: 'linear',
  selectedXAxisMetricKey: '',
  xAxisKey: RunsChartsLineChartXAxisType.STEP,
};

describe('RunsChartsConfigureModal', () => {
  const renderTestComponent = (onSubmit?: () => void) => {
    render(
      <RunsChartsConfigureModal
        config={sampleLineChartConfig}
        chartRunData={sampleChartData}
        groupBy={null}
        metricKeyList={[]}
        onCancel={() => {}}
        onSubmit={onSubmit ?? (() => {})}
        paramKeyList={[]}
      />,
      {
        wrapper: ({ children }) => (
          <DesignSystemProvider>
            <TestApolloProvider>
              <MockedReduxStoreProvider
                state={{
                  entities: {
                    metricsByRunUuid: {},
                  },
                }}
              >
                <IntlProvider locale="en">{children}</IntlProvider>
              </MockedReduxStoreProvider>
            </TestApolloProvider>
          </DesignSystemProvider>
        ),
      },
    );
  };

  const getLastPropsForRunsMetricsLinePlot = () => last(jest.mocked(RunsMetricsLinePlot).mock.calls)?.[0];

  test('it should render line plot configuration form and preview with the correct props', async () => {
    const onSubmit = jest.fn();
    renderTestComponent(onSubmit);

    await waitFor(() => {
      expect(screen.getByText('Edit chart')).toBeInTheDocument();
    });

    expect(getLastPropsForRunsMetricsLinePlot()).toEqual(
      expect.objectContaining({
        displayPoints: undefined,
      }),
    );

    expect(screen.getByLabelText('Display points: Auto')).toBeChecked();

    await userEvent.click(screen.getByLabelText('Display points: On'));

    expect(getLastPropsForRunsMetricsLinePlot()).toEqual(
      expect.objectContaining({
        displayPoints: true,
      }),
    );

    await userEvent.click(screen.getByLabelText('Display points: Off'));

    expect(getLastPropsForRunsMetricsLinePlot()).toEqual(
      expect.objectContaining({
        displayPoints: false,
      }),
    );

    await userEvent.click(screen.getByLabelText('Display points: Auto'));

    expect(getLastPropsForRunsMetricsLinePlot()).toEqual(
      expect.objectContaining({
        displayPoints: undefined,
      }),
    );

    await userEvent.click(screen.getByLabelText('Display points: On'));
    await userEvent.click(screen.getByText('Save changes'));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        displayPoints: true,
      }),
    );
  });
});
