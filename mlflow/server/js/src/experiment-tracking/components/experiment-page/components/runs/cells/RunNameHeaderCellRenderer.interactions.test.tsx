import { render, screen, fireEvent } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { RunNameHeaderCellRenderer } from './RunNameHeaderCellRenderer';

// Create mock state utility
const createMockState = () => ({
  hideFinishedRuns: false,
  runLimit: null as number | null,
  searchFilter: '',
  orderByKey: '',
  orderByAsc: true,
  startTime: 'ALL',
  lifecycleFilter: 'ACTIVE',
  datasetsFilter: [],
  modelVersionFilter: 'All Runs',
});

// Store the onValueChange and onCheckedChange for testing
let mockOnValueChange: ((value: string) => void) | null = null;
let mockOnCheckedChange: ((checked: boolean) => void) | null = null;

// Mock the hooks with correct return signatures
const mockUpdateSearchFacets = jest.fn();
const mockSearchFacetsState = createMockState();

jest.mock('../../../hooks/useExperimentPageSearchFacets', () => ({
  useExperimentPageSearchFacets: () => [mockSearchFacetsState, ['123'], false],
  useUpdateExperimentPageSearchFacets: () => mockUpdateSearchFacets,
}));

// Mock design system components
jest.mock('@databricks/design-system', () => ({
  useDesignSystemTheme: () => ({
    theme: {
      spacing: { xs: 8, sm: 16 },
      colors: {
        textSecondary: '#666',
        actionTertiaryTextHover: '#333',
      },
    },
  }),
  SortAscendingIcon: () => <div data-testid="sort-ascending-icon" />,
  SortDescendingIcon: () => <div data-testid="sort-descending-icon" />,
  VisibleOffIcon: () => <div data-testid="visible-off-icon" />,
  Icon: ({ component: Component }: { component: any }) => <Component data-testid="visible-icon" />,
  Tooltip: ({ children, content, componentId }: any) => (
    <div data-testid="tooltip" data-component-id={componentId} title={content}>
      {children}
    </div>
  ),
  Button: ({ children, onClick, icon, 'aria-label': ariaLabel, componentId }: any) => (
    <button
      onClick={onClick}
      aria-label={ariaLabel}
      data-component-id={componentId}
      data-testid="visibility-toggle-button"
    >
      {icon}
      {children}
    </button>
  ),
  DropdownMenu: {
    Root: ({ children }: any) => <div data-testid="dropdown-root">{children}</div>,
    Trigger: ({ children, asChild }: any) => (asChild ? children : <div>{children}</div>),
    Content: ({ children }: any) => <div data-testid="dropdown-content">{children}</div>,
    RadioGroup: ({ children, value, onValueChange, componentId }: any) => {
      mockOnValueChange = onValueChange;
      return (
        <div data-testid="radio-group" data-component-id={componentId} data-value={value}>
          {children}
        </div>
      );
    },
    RadioItem: ({ children, value }: any) => (
      <div
        data-testid={`radio-item-${value}`}
        onClick={() => mockOnValueChange?.(value)}
        role="menuitemradio"
        aria-checked="false"
      >
        {children}
      </div>
    ),
    CheckboxItem: ({ children, checked, onCheckedChange, componentId }: any) => {
      mockOnCheckedChange = onCheckedChange;
      return (
        <div
          data-testid="checkbox-item"
          data-component-id={componentId}
          data-checked={checked}
          onClick={() => mockOnCheckedChange?.(!checked)}
          role="menuitemcheckbox"
          aria-checked={checked ? 'true' : 'false'}
        >
          {children}
        </div>
      );
    },
    ItemIndicator: () => <div data-testid="item-indicator" />,
    Separator: () => <div data-testid="separator" />,
  },
}));

// Mock the SVG icon
jest.mock('../../../../../../common/static/icon-visible-fill.svg', () => ({
  ReactComponent: () => <div data-testid="visible-fill-icon" />,
}));

const renderComponent = (props = {}) => {
  return render(
    <IntlProvider locale="en">
      <RunNameHeaderCellRenderer
        displayName="Runs"
        enableSorting
        context={{
          orderByKey: 'metrics.accuracy',
          orderByAsc: false,
        }}
        {...props}
      />
    </IntlProvider>,
  );
};

describe('RunNameHeaderCellRenderer - Interactions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Object.assign(mockSearchFacetsState, {
      hideFinishedRuns: false,
      runLimit: null,
    });
    // Reset mock callbacks
    mockOnValueChange = null;
    mockOnCheckedChange = null;
  });

  describe('Run Limit Selection', () => {
    const runLimitTestCases = [
      { name: 'selects 10 runs', testId: 'radio-item-10', expectedCall: { runLimit: 10 } },
      { name: 'selects 20 runs', testId: 'radio-item-20', expectedCall: { runLimit: 20 } },
      { name: 'selects all runs', testId: 'radio-item-all', expectedCall: { runLimit: null } },
    ];

    test.each(runLimitTestCases)('$name', ({ testId, expectedCall }) => {
      renderComponent();

      fireEvent.click(screen.getByTestId(testId));

      expect(mockUpdateSearchFacets).toHaveBeenCalledWith(expectedCall);
    });
  });

  describe('Toggle Finished Runs', () => {
    const toggleTestCases = [
      {
        name: 'enables hiding when currently showing all',
        initialState: false,
        expectedCall: { hideFinishedRuns: true },
      },
      {
        name: 'disables hiding when currently hiding',
        initialState: true,
        expectedCall: { hideFinishedRuns: false },
      },
    ];

    test.each(toggleTestCases)('$name', ({ initialState, expectedCall }) => {
      (mockSearchFacetsState as any).hideFinishedRuns = initialState;
      renderComponent();

      const checkbox = screen.getByTestId('checkbox-item');
      fireEvent.click(checkbox);

      expect(mockUpdateSearchFacets).toHaveBeenCalledWith(expectedCall);
    });
  });

  describe('Sorting Interactions', () => {
    const sortingInteractionTestCases = [
      {
        name: 'toggles sort order for same column',
        currentKey: 'metrics.accuracy',
        columnKey: 'metrics.accuracy',
        currentAsc: true,
        expectedCall: { orderByKey: 'metrics.accuracy', orderByAsc: false },
      },
      {
        name: 'resets to descending for new column',
        currentKey: 'metrics.accuracy',
        columnKey: 'metrics.loss',
        currentAsc: true,
        expectedCall: { orderByKey: 'metrics.loss', orderByAsc: false },
      },
    ];

    test.each(sortingInteractionTestCases)('$name', ({ currentKey, columnKey, currentAsc, expectedCall }) => {
      const mockColumn = {
        getColDef: () => ({
          headerComponentParams: { canonicalSortKey: columnKey },
        }),
      };

      renderComponent({
        column: mockColumn,
        context: { orderByKey: currentKey, orderByAsc: currentAsc },
      });

      fireEvent.click(screen.getByTestId('sort-header-Runs'));

      expect(mockUpdateSearchFacets).toHaveBeenCalledWith(expectedCall);
    });

    it('does not call updateSearchFacets when sorting is disabled', () => {
      const mockColumn = {
        getColDef: () => ({
          headerComponentParams: { canonicalSortKey: 'metrics.accuracy' },
        }),
      };

      renderComponent({
        column: mockColumn,
        enableSorting: false,
        context: { orderByKey: 'metrics.accuracy', orderByAsc: true },
      });

      fireEvent.click(screen.getByTestId('sort-header-Runs'));

      expect(mockUpdateSearchFacets).not.toHaveBeenCalled();
    });
  });

  describe('State Change Re-renders', () => {
    it('updates icon display after hideFinishedRuns state change', () => {
      (mockSearchFacetsState as any).hideFinishedRuns = false;
      const { rerender } = renderComponent();

      expect(screen.getByTestId('visible-fill-icon')).toBeInTheDocument();

      const checkbox = screen.getByTestId('checkbox-item');
      fireEvent.click(checkbox);
      expect(mockUpdateSearchFacets).toHaveBeenCalledWith({ hideFinishedRuns: true });

      // Simulate state update
      (mockSearchFacetsState as any).hideFinishedRuns = true;
      rerender(
        <IntlProvider locale="en">
          <RunNameHeaderCellRenderer
            displayName="Runs"
            enableSorting
            context={{
              orderByKey: 'metrics.accuracy',
              orderByAsc: false,
            }}
          />
        </IntlProvider>,
      );

      expect(screen.getByTestId('visible-off-icon')).toBeInTheDocument();
      expect(screen.queryByTestId('visible-fill-icon')).not.toBeInTheDocument();
    });

    it('updates radio group selection after runLimit state change', () => {
      (mockSearchFacetsState as any).runLimit = null;
      const { rerender } = renderComponent();

      expect(screen.getByTestId('radio-group')).toHaveAttribute('data-value', 'all');

      fireEvent.click(screen.getByTestId('radio-item-10'));
      expect(mockUpdateSearchFacets).toHaveBeenCalledWith({ runLimit: 10 });

      // Simulate state update
      (mockSearchFacetsState as any).runLimit = 10;
      rerender(
        <IntlProvider locale="en">
          <RunNameHeaderCellRenderer
            displayName="Runs"
            enableSorting
            context={{
              orderByKey: 'metrics.accuracy',
              orderByAsc: false,
            }}
          />
        </IntlProvider>,
      );

      expect(screen.getByTestId('radio-group')).toHaveAttribute('data-value', '10');
    });
  });

  describe('Dropdown Trigger Interaction', () => {
    it('renders dropdown trigger button that can be clicked', () => {
      renderComponent();

      const triggerButton = screen.getByTestId('visibility-toggle-button');
      expect(triggerButton).toBeInTheDocument();

      // Should be clickable without errors
      expect(() => fireEvent.click(triggerButton)).not.toThrow();
    });

    it('has proper accessibility attributes on trigger button', () => {
      renderComponent();

      const triggerButton = screen.getByTestId('visibility-toggle-button');
      expect(triggerButton).toHaveAttribute('aria-label', 'Toggle visibility of runs');
      expect(triggerButton).toHaveAttribute('data-component-id', 'run_name_header_visibility_dropdown');
    });
  });
});
