import { render, screen } from '@testing-library/react';
import { CellDifference } from './DifferenceViewPlot.utils';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsDifferenceCardConfig } from '../../runs-charts.types';
import { DifferenceCardConfigCompareGroup, RunsChartType } from '../../runs-charts.types';
import { IntlProvider } from 'react-intl';
import { DifferenceChartCellDirection } from '../../utils/differenceView';
import { DifferenceViewPlot } from './DifferenceViewPlot';

describe('DifferenceViewPlot', () => {
  const originalGetBoundingClientRect = window.Element.prototype.getBoundingClientRect;

  // Required to see virtualized entities
  beforeAll(() => {
    window.Element.prototype.getBoundingClientRect = () => ({ height: 300, width: 100 } as DOMRect);
  });
  afterAll(() => {
    window.Element.prototype.getBoundingClientRect = originalGetBoundingClientRect;
  });

  const previewData: RunsChartsRunData[] = [
    {
      uuid: 'run1-uuid',
      displayName: 'run1',
      images: {},
      runInfo: {
        runUuid: 'run1-uuid',
        experimentId: 'experiment1-uuid',
        status: 'FINISHED',
        startTime: 1633027589000,
        endTime: 1633027589000,
        lifecycleStage: 'active',
        runName: 'run1',
        artifactUri: 'artifactUri',
      },
      metrics: {
        metric1: { key: 'metric1', value: 10, step: 1, timestamp: 1633027589000 },
        metric2: { key: 'metric2', value: 20, step: 2, timestamp: 1633027589000 },
      },
      params: {
        param1: { key: 'key1', value: 'value1' },
        param2: { key: 'key2', value: 'value2' },
      },
      tags: {
        tag1: { key: 'tag1', value: 'value1' },
        tag2: { key: 'tag2', value: 'value2' },
      },
    },
    {
      uuid: 'run2-uuid',
      displayName: 'run2',
      runInfo: {
        runUuid: 'run2-uuid',
        experimentId: 'experiment1-uuid',
        status: 'FINISHED',
        startTime: 1633027589000,
        endTime: 1633027589000,
        lifecycleStage: 'active',
        runName: 'run2',
        artifactUri: 'artifact',
      },
      images: {},
      metrics: {
        metric1: { key: 'metric1', value: 30, step: 3, timestamp: 1633027589000 },
        metric2: { key: 'metric2', value: 40, step: 4, timestamp: 1633027589000 },
      },
      params: {
        param1: { key: 'param1', value: 'value3' },
        param2: { key: 'param2', value: 'value4' },
      },
      tags: {
        tag1: { key: 'tag1', value: 'value3' },
        tag2: { key: 'tag2', value: 'value4' },
      },
    },
  ];

  const cardConfig: RunsChartsDifferenceCardConfig = {
    type: RunsChartType.DIFFERENCE,
    compareGroups: [DifferenceCardConfigCompareGroup.MODEL_METRICS, DifferenceCardConfigCompareGroup.PARAMETERS],
    chartName: 'Runs difference view',
    showChangeFromBaseline: true,
    showDifferencesOnly: true,
    baselineColumnUuid: '',
    deleted: false,
    isGenerated: false,
  };

  const groupBy = null;

  const setCardConfig = jest.fn();

  it('renders the DifferenceViewPlot component', async () => {
    render(
      <IntlProvider locale="en">
        <DifferenceViewPlot
          previewData={previewData}
          cardConfig={cardConfig}
          groupBy={groupBy}
          setCardConfig={setCardConfig}
        />
      </IntlProvider>,
    );

    // Assert that the component is rendered
    expect(screen.getByText('Compare by')).toBeInTheDocument();

    // Assert that the selected groups exists
    expect(screen.getByText('Model Metrics')).toBeInTheDocument();
    expect(screen.getByText('Parameters')).toBeInTheDocument();
    expect(screen.queryByText('Tags')).not.toBeInTheDocument();
    expect(screen.queryByText('Attributes')).not.toBeInTheDocument();
    expect(screen.queryByText('System Metrics')).not.toBeInTheDocument();
  });
});

describe('CellDifference', () => {
  it('renders the CellDifference component', () => {
    render(
      <IntlProvider locale="en">
        <CellDifference label="10" direction={DifferenceChartCellDirection.POSITIVE} />
      </IntlProvider>,
    );

    // Assert that the component is rendered
    expect(screen.getByText('10')).toBeInTheDocument();
    expect(screen.getByTestId('positive-cell-direction')).toBeInTheDocument();
  });

  it('renders the CellDifference component with negative direction', () => {
    render(
      <IntlProvider locale="en">
        <CellDifference label="-10" direction={DifferenceChartCellDirection.NEGATIVE} />
      </IntlProvider>,
    );

    // Assert that the component is rendered
    expect(screen.getByText('-10')).toBeInTheDocument();
    expect(screen.getByTestId('negative-cell-direction')).toBeInTheDocument();
  });

  it('renders the CellDifference component with same direction', () => {
    render(
      <IntlProvider locale="en">
        <CellDifference label="0" direction={DifferenceChartCellDirection.SAME} />
      </IntlProvider>,
    );

    // Assert that the component is rendered
    expect(screen.getByText('0')).toBeInTheDocument();
    expect(screen.getByTestId('same-cell-direction')).toBeInTheDocument();
  });
});
