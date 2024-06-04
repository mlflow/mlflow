import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event-14';
import { renderWithIntl, screen } from '../../../../../common/utils/TestUtils.react18';
import { ExperimentViewTracesControls } from './ExperimentViewTracesControls';
import { ExperimentViewTracesTableColumns } from './ExperimentViewTraces.utils';

// Disable pointer events check for DialogCombobox which masks the elements we want to click
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

describe('ExperimentViewTracesControls', () => {
  const mockOnChangeFilter = jest.fn();
  const mockToggleHiddenColumn = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders the component with initial filter value', () => {
    const filter = 'test-filter';
    renderWithIntl(
      <ExperimentViewTracesControls
        filter={filter}
        onChangeFilter={mockOnChangeFilter}
        toggleHiddenColumn={mockToggleHiddenColumn}
      />,
    );

    const filterInput = screen.getByPlaceholderText('Search traces');
    expect(filterInput).toBeInTheDocument();
    expect(filterInput).toHaveValue(filter);
  });

  test('calls onChangeFilter when filter input value changes', async () => {
    const filter = 'test-filter';
    renderWithIntl(
      <ExperimentViewTracesControls
        filter={filter}
        onChangeFilter={mockOnChangeFilter}
        toggleHiddenColumn={mockToggleHiddenColumn}
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
      <ExperimentViewTracesControls
        filter={filter}
        onChangeFilter={mockOnChangeFilter}
        toggleHiddenColumn={mockToggleHiddenColumn}
      />,
    );

    const clearButton = screen.getByRole('button', { name: 'close-circle' });
    await userEvent.click(clearButton);

    expect(mockOnChangeFilter).toHaveBeenCalledWith('');
  });

  test('calls toggleHiddenColumn when a column checkbox is clicked', async () => {
    const filter = 'test-filter';

    // Initially, 'Execution time' column is hidden
    const hiddenColumns = [ExperimentViewTracesTableColumns.latency];
    renderWithIntl(
      <ExperimentViewTracesControls
        filter={filter}
        hiddenColumns={hiddenColumns}
        onChangeFilter={mockOnChangeFilter}
        toggleHiddenColumn={mockToggleHiddenColumn}
      />,
    );

    // Open the column dropdown
    await userEvent.click(screen.getByRole('combobox', { name: 'Columns' }));

    // Confirm that the 'Execution time' column is hidden
    expect(screen.getByRole('checkbox', { name: 'Execution time' })).not.toBeChecked();

    // Click the 'Tags' column checkbox
    await userEvent.click(screen.getByRole('checkbox', { name: 'Tags' }));

    // Confirm that the 'Tags' column checkbox is checked
    expect(mockToggleHiddenColumn).toHaveBeenCalledWith('tags'); // Replace 'column1' with the actual column key
  });
});
