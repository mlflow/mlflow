import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { renderWithIntl, screen } from '../../../common/utils/TestUtils.react18';
import { TracesViewControls } from './TracesViewControls';

// Disable pointer events check for DialogCombobox which masks the elements we want to click
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

describe('ExperimentViewTracesControls', () => {
  const mockOnChangeFilter = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders the component with initial filter value', () => {
    const filter = 'test-filter';
    renderWithIntl(
      <TracesViewControls
        experimentIds={['0']}
        filter={filter}
        onChangeFilter={mockOnChangeFilter}
        rowSelection={{}}
        setRowSelection={() => {}}
        refreshTraces={() => {}}
        baseComponentId="test"
        traces={[]}
      />,
    );

    const filterInput = screen.getByPlaceholderText('Search traces');
    expect(filterInput).toBeInTheDocument();
    expect(filterInput).toHaveValue(filter);
  });

  test('calls onChangeFilter when filter input value changes', async () => {
    const filter = 'test-filter';
    renderWithIntl(
      <TracesViewControls
        experimentIds={['0']}
        filter={filter}
        onChangeFilter={mockOnChangeFilter}
        rowSelection={{}}
        setRowSelection={() => {}}
        refreshTraces={() => {}}
        baseComponentId="test"
        traces={[]}
      />,
    );

    const filterInput = screen.getByPlaceholderText('Search traces');
    const newFilterValue = 'new-filter';
    await userEvent.clear(filterInput);
    await userEvent.type(filterInput, `${newFilterValue}{enter}`);

    expect(mockOnChangeFilter).toHaveBeenCalledWith(newFilterValue);
  });

  test('calls onChangeFilter with empty string when clear button is clicked', async () => {
    const filter = 'test-filter';
    renderWithIntl(
      <TracesViewControls
        experimentIds={['0']}
        filter={filter}
        onChangeFilter={mockOnChangeFilter}
        rowSelection={{}}
        setRowSelection={() => {}}
        refreshTraces={() => {}}
        baseComponentId="test"
        traces={[]}
      />,
    );

    const clearButton = screen.getByRole('button', { name: 'close-circle' });
    await userEvent.click(clearButton);

    expect(mockOnChangeFilter).toHaveBeenCalledWith('');
  });
});
