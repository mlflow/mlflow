import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { RunsChartsTooltipBody } from './RunsChartsTooltipBody';
import { RunsChartsTooltipMode } from '../hooks/useRunsChartsTooltip';
import { RunsChartType } from '../runs-charts.types';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import type { RunsChartsRunData } from './RunsCharts.common';
import type { RunsChartsBarCardConfig } from '../runs-charts.types';
import type { RunsMetricsSingleTraceTooltipData } from './RunsMetricsLinePlot';

jest.mock('../../experiment-page/hooks/useExperimentIds', () => ({
  useExperimentIds: () => ['test-experiment-id'],
}));

const baseRun: RunsChartsRunData = {
  uuid: 'run-uuid-1',
  displayName: 'test-run',
  metrics: { 'test-metric': { key: 'test-metric', value: 0.9, step: 1, timestamp: 0 } },
  params: {},
  tags: {},
  images: {},
  color: '#1f77b4',
  pinned: false,
  pinnable: true,
  hidden: false,
};

const barChartConfig: RunsChartsBarCardConfig = {
  type: RunsChartType.BAR,
  uuid: 'chart-uuid',
  deleted: false,
  isGenerated: false,
  metricKey: 'test-metric',
};

const mockHoverData: RunsMetricsSingleTraceTooltipData = {
  xValue: 1,
  yValue: 0.9,
  index: 0,
  label: 'Step',
};

const renderTooltip = (run: RunsChartsRunData, isHovering = true) => {
  render(
    <DesignSystemProvider>
      <IntlProvider locale="en">
        <MemoryRouter>
          <RunsChartsTooltipBody
            runUuid={run.uuid}
            isHovering={isHovering}
            mode={RunsChartsTooltipMode.Simple}
            hoverData={mockHoverData}
            chartData={barChartConfig}
            contextData={{ runs: [run], onTogglePin: jest.fn(), onHideRun: jest.fn() }}
            closeContextMenu={jest.fn()}
          />
        </MemoryRouter>
      </IntlProvider>
    </DesignSystemProvider>,
  );
};

describe('RunsChartsTooltipBody', () => {
  it('shows description when run has a description tag', () => {
    const runWithDescription: RunsChartsRunData = {
      ...baseRun,
      tags: {
        [NOTE_CONTENT_TAG]: { key: NOTE_CONTENT_TAG, value: 'This is a test run description.' },
      },
    };
    renderTooltip(runWithDescription);
    expect(screen.getByText('This is a test run description.')).toBeInTheDocument();
  });

  it('does not show description section when run has no description', () => {
    renderTooltip(baseRun);
    expect(screen.queryByText('This is a test run description.')).not.toBeInTheDocument();
  });

  it('shows description in context menu mode (isHovering=false)', () => {
    const runWithDescription: RunsChartsRunData = {
      ...baseRun,
      tags: {
        [NOTE_CONTENT_TAG]: { key: NOTE_CONTENT_TAG, value: 'Context menu description.' },
      },
    };
    renderTooltip(runWithDescription, false);
    expect(screen.getByText('Context menu description.')).toBeInTheDocument();
  });

  it('shows run name in the tooltip', () => {
    renderTooltip(baseRun);
    expect(screen.getByText('test-run')).toBeInTheDocument();
  });
});
