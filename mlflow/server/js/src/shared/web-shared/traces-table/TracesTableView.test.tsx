import { describe, expect, test, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import { TracesTableView, type TracesTableViewProps, type TracesTableViewState } from './TracesTableView';
import { makeTraces } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';

const baseProps = (over: Partial<TracesTableViewProps> = {}): TracesTableViewProps => ({
  viewState: 'ready',
  traces: makeTraces(2),
  visibleColumns: ['start_time', 'input'],
  initialColumnSizing: {},
  onColumnSizingSettled: jest.fn(),
  isLoading: false,
  isFetching: false,
  skeletonRowCount: 25,
  onTraceSelected: jest.fn(),
  selectedForBulk: new Map(),
  isAllOnPageSelected: false,
  isSomeOnPageSelected: false,
  onToggleBulkRow: jest.fn(),
  onToggleBulkAll: jest.fn(),
  sort: 'start_time',
  dir: 'desc',
  onSort: jest.fn(),
  searchValue: '',
  onSearchChange: jest.fn(),
  onSearchClear: jest.fn(),
  pageIndex: 1,
  pageSize: 25,
  onPageChange: jest.fn(),
  onPageSizeChange: jest.fn(),
  hasNext: false,
  hasPrev: false,
  onClearFilters: jest.fn(),
  onRetry: jest.fn(),
  onHideColumn: jest.fn(),
  ...over,
});

describe('TracesTableView', () => {
  test('always renders the toolbar search box', async () => {
    await renderWithProviders(<TracesTableView {...baseProps({ viewState: 'empty' })} />);
    expect(screen.getByRole('textbox', { name: /Search traces/ })).toBeInTheDocument();
  });

  test('renders the toolbar control slots in order around the search', async () => {
    await renderWithProviders(
      <TracesTableView
        {...baseProps({
          leftControls: <button type="button">left-slot</button>,
          rightControls: <button type="button">right-slot</button>,
        })}
      />,
    );
    const left = screen.getByRole('button', { name: 'left-slot' });
    const search = screen.getByRole('textbox', { name: /Search traces/ });
    const right = screen.getByRole('button', { name: 'right-slot' });
    expect(left.compareDocumentPosition(search) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(search.compareDocumentPosition(right) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  test('renders the banner slot above the region', async () => {
    await renderWithProviders(<TracesTableView {...baseProps({ bannerSlot: <div>banner-content</div> })} />);
    expect(screen.getByText('banner-content')).toBeInTheDocument();
  });

  test('renders searchSuffix inside the search input', async () => {
    await renderWithProviders(
      <TracesTableView {...baseProps({ searchSuffix: <button type="button">suffix-slot</button> })} />,
    );
    expect(screen.getByRole('button', { name: 'suffix-slot' })).toBeInTheDocument();
  });

  test.each<{ viewState: TracesTableViewState; expected: RegExp }>([
    { viewState: 'ready', expected: /Rows per page/ },
    { viewState: 'empty', expected: /No traces yet/ },
    { viewState: 'no-results', expected: /No traces match your filters/ },
    { viewState: 'no-more-results', expected: /No more results/ },
    { viewState: 'error', expected: /Couldn't load traces/ },
  ])('viewState=$viewState renders the matching region', async ({ viewState, expected }) => {
    await renderWithProviders(<TracesTableView {...baseProps({ viewState })} />);
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  test('the ready state renders the table (a column header) and pagination', async () => {
    await renderWithProviders(<TracesTableView {...baseProps({ viewState: 'ready' })} />);
    expect(screen.getByRole('columnheader', { name: /Time/ })).toBeInTheDocument();
    expect(screen.getByText(/Rows per page/)).toBeInTheDocument();
  });

  test('customEmptyState short-circuits the viewState region but keeps the toolbar', async () => {
    await renderWithProviders(
      <TracesTableView {...baseProps({ viewState: 'empty', customEmptyState: <div>pick-a-warehouse</div> })} />,
    );
    expect(screen.getByText('pick-a-warehouse')).toBeInTheDocument();
    // The built-in empty state is NOT rendered when a custom one is provided.
    expect(screen.queryByText(/No traces yet/)).not.toBeInTheDocument();
    // The toolbar still renders.
    expect(screen.getByRole('textbox', { name: /Search traces/ })).toBeInTheDocument();
  });

  test('no-more-results keeps the pagination bar so the user can step back', async () => {
    await renderWithProviders(<TracesTableView {...baseProps({ viewState: 'no-more-results', hasPrev: true })} />);
    expect(screen.getByText(/No more results/)).toBeInTheDocument();
    expect(screen.getByText(/Rows per page/)).toBeInTheDocument();
  });

  test('wraps the pagination bar in PaginationBarWrapper when provided', async () => {
    // The consumer (MLflow) passes AssistantAwareActionBar so the floating Assistant button clears
    // the pinned bar. Assert the wrapper actually encloses the bar (not just renders alongside it).
    const PaginationBarWrapper = ({ children }: { children: React.ReactNode }) => (
      <div data-testid="pagination-wrapper">{children}</div>
    );
    await renderWithProviders(<TracesTableView {...baseProps({ viewState: 'ready', PaginationBarWrapper })} />);
    const wrapper = screen.getByTestId('pagination-wrapper');
    expect(wrapper).toContainElement(screen.getByText(/Rows per page/));
  });
});
