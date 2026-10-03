import { describe, expect, jest, test } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { render, screen } from '../../../../../../common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { ATTRIBUTE_COLUMN_SORT_KEY } from '../../../../../constants';
import { RUNS_VISIBILITY_MODE } from '../../../models/ExperimentPageUIState';
import { ExperimentViewRunsTableHeaderContextProvider } from '../ExperimentViewRunsTableHeaderContext';
import type { Column, ColumnApi } from '@ag-grid-community/core';
import type { ColumnHeaderCellProps } from './ColumnHeaderCell';
import { ColumnHeaderCell } from './ColumnHeaderCell';
import { DateCellRenderer } from './DateCellRenderer';

const mockUpdateSearchFacets = jest.fn();

jest.mock('../../../hooks/useExperimentPageSearchFacets', () => ({
  useUpdateExperimentPageSearchFacets: () => mockUpdateSearchFacets,
}));

const startTime = Date.UTC(2024, 0, 1, 12, 0, 0);
const value = {
  startTime,
  referenceTime: new Date(startTime + 60_000),
  runStatus: 'FINISHED',
  experimentId: '0',
  runUuid: 'run',
  isParent: false,
  hasExpander: false,
  belongsToGroup: false,
  level: 0,
};

const renderTable = (headerProps: Partial<ColumnHeaderCellProps> = {}) =>
  render(
    <IntlProvider locale="en" timeZone="UTC">
      <DesignSystemProvider>
        <ExperimentViewRunsTableHeaderContextProvider runsHiddenMode={RUNS_VISIBILITY_MODE.FIRST_10_RUNS}>
          <ColumnHeaderCell
            enableSorting
            canonicalSortKey={ATTRIBUTE_COLUMN_SORT_KEY.DATE}
            displayName="Created"
            context={{ orderByKey: ATTRIBUTE_COLUMN_SORT_KEY.DATE, orderByAsc: false }}
            {...headerProps}
          />
          <DateCellRenderer value={value} />
          <DateCellRenderer value={{ ...value, startTime: startTime + 1_000 }} />
        </ExperimentViewRunsTableHeaderContextProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('run timestamp display', () => {
  test('toggles every date cell between relative and absolute timestamps', async () => {
    renderTable();
    expect(screen.getByText('1 minute ago')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Show absolute timestamps' })).toHaveAttribute('aria-pressed', 'false');
    mockUpdateSearchFacets.mockClear();
    await userEvent.click(screen.getByRole('button', { name: 'Show absolute timestamps' }));
    expect(screen.getByText('01/01/2024, 12:00:00 PM')).toBeInTheDocument();
    expect(screen.getByText('01/01/2024, 12:00:01 PM')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Show relative timestamps' })).toHaveAttribute('aria-pressed', 'true');
    await userEvent.click(screen.getByRole('button', { name: 'Show relative timestamps' }));
    expect(screen.getByText('1 minute ago')).toBeInTheDocument();
    expect(screen.queryByText('01/01/2024, 12:00:00 PM')).not.toBeInTheDocument();
    expect(mockUpdateSearchFacets).not.toHaveBeenCalled();
  });

  test('keeps the absolute timestamp in the tooltip by default', () => {
    renderTable();
    expect(screen.getByText('1 minute ago').closest('span')).toHaveAttribute('title', '01/01/2024, 12:00:00 PM');
  });

  test('does not offer timestamp formatting for other column headers', () => {
    renderTable({ canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.RUN_NAME });
    expect(screen.queryByRole('button', { name: 'Show absolute timestamps' })).not.toBeInTheDocument();
  });

  test.each([150, 400])('makes absolute timestamps visible without shrinking a %s px column', async (width) => {
    const column = { getActualWidth: () => width } as Column;
    const setColumnWidth = jest.fn();
    const columnApi = { setColumnWidth } as unknown as ColumnApi;
    renderTable({ column, columnApi });
    await userEvent.click(screen.getByRole('button', { name: 'Show absolute timestamps' }));
    expect(setColumnWidth).toHaveBeenCalledWith(column, Math.max(width, 240));
    await userEvent.click(screen.getByRole('button', { name: 'Show relative timestamps' }));
    expect(setColumnWidth).toHaveBeenCalledTimes(1);
  });
});
