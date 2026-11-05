import { describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { IntlProvider } from '@databricks/i18n';

import { CategoricalAggregateChart } from './CategoricalAggregateChart';
import type { AssessmentAggregates, AssessmentInfo } from '../../types';
import { getBarChartData, type StackedBarchartItem } from '../../utils/AggregationUtils';

jest.mock('../../utils/AggregationUtils', () => ({
  ERROR_KEY: 'Error',
  getBarChartData: jest.fn(),
}));

jest.mock('../../utils/DisplayUtils', () => ({
  getDisplayScore: (_info: unknown, fraction: number) => `${Math.round(fraction * 100)}%`,
  getDisplayScoreChange: jest.fn(),
}));

const theme = {
  colors: {
    actionDefaultBackgroundHover: '#f0f0f0',
    actionDefaultBackgroundPress: '#e0e0e0',
    backgroundSecondary: '#f7f7f7',
    textSecondary: '#666',
    textValidationWarning: '#ff6b6b',
  },
  spacing: { xs: 8, sm: 8, xl: 32 },
  typography: { fontSizeSm: 12, typographyBoldFontWeight: 600 },
  general: { borderRadiusBase: 4 },
} as any;

const intl = { formatMessage: (message: { defaultMessage?: string }) => message.defaultMessage ?? '' } as any;

const assessmentInfo = {
  name: 'correctness',
  displayName: 'Correctness',
  dtype: 'pass-fail',
  uniqueValues: new Set(['yes', 'no']),
} as unknown as AssessmentInfo;

const bar = (name: string, value: number, fraction: number, toggleFilter = jest.fn()): StackedBarchartItem => ({
  name,
  current: { value, fraction, isSelected: false, toggleFilter, tooltip: '' },
  backgroundColor: '#0f0',
});

const renderChart = (layout?: 'stacked' | 'inline') =>
  render(
    <IntlProvider locale="en">
      <CategoricalAggregateChart
        theme={theme}
        intl={intl}
        assessmentInfo={assessmentInfo}
        assessmentAggregates={{} as AssessmentAggregates}
        allAssessmentFilters={[]}
        toggleAssessmentFilter={jest.fn()}
        layout={layout}
      />
    </IntlProvider>,
  );

describe('CategoricalAggregateChart layouts', () => {
  it('keeps the stacked layout, with no inline bars, when no layout is given', () => {
    jest.mocked(getBarChartData).mockReturnValue([bar('null', 1, 0.1), bar('Pass', 6, 0.6), bar('Fail', 3, 0.3)]);
    renderChart();

    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    // Pass/fail rows keep their original order in the stacked layout.
    expect(screen.getAllByRole('row').map((row) => row.textContent)).toEqual(['null1', 'Pass60%', 'Fail30%']);
  });

  it('puts each value on one line with a bar, real values first and missing ones last', () => {
    jest.mocked(getBarChartData).mockReturnValue([bar('null', 1, 0.1), bar('Pass', 6, 0.6), bar('Fail', 3, 0.3)]);
    renderChart('inline');

    expect(screen.getAllByRole('row').map((row) => row.textContent)).toEqual(['Pass60%', 'Fail30%', 'null1']);
    expect(screen.getByRole('progressbar', { name: 'Pass' })).toHaveAttribute('aria-valuenow', '60');
    expect(screen.getByRole('progressbar', { name: 'Fail' })).toHaveAttribute('aria-valuenow', '30');
    // Missing values show a count and no bar.
    expect(screen.queryByRole('progressbar', { name: 'null' })).not.toBeInTheDocument();
  });

  it('toggles the clicked value’s filter in the inline layout', async () => {
    const toggleFilter = jest.fn();
    jest.mocked(getBarChartData).mockReturnValue([bar('Pass', 6, 0.6, toggleFilter), bar('Fail', 3, 0.3)]);
    renderChart('inline');

    await userEvent.click(screen.getByRole('progressbar', { name: 'Pass' }));

    expect(toggleFilter).toHaveBeenCalledTimes(1);
  });
});
