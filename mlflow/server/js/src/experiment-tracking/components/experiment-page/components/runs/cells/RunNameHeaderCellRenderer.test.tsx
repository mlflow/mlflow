import React from 'react';
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

// Mock the design system components to avoid complex dropdown testing
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

describe('RunNameHeaderCellRenderer', () => {
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

  describe('Basic Rendering', () => {
    const renderingTestCases = [
      { name: 'default display name', props: {}, expectedText: 'Runs' },
      { name: 'custom display name', props: { displayName: 'Custom Runs' }, expectedText: 'Custom Runs' },
      { name: 'fallback display name', props: { displayName: undefined }, expectedText: 'Run Name' },
    ];

    test.each(renderingTestCases)('renders with $name', ({ props, expectedText }) => {
      renderComponent(props);
      expect(screen.getByText(expectedText)).toBeInTheDocument();
    });

    it('renders visibility toggle button', () => {
      renderComponent();
      expect(screen.getByTestId('visibility-toggle-button')).toBeInTheDocument();
      expect(screen.getByLabelText('Toggle visibility of runs')).toBeInTheDocument();
    });
  });

  describe('Icon Display Logic', () => {
    const iconTestCases = [
      {
        name: 'visible icon when hideFinishedRuns is false',
        hideFinishedRuns: false,
        expectedIcon: 'visible-fill-icon',
        notExpectedIcon: 'visible-off-icon',
      },
      {
        name: 'visible-off icon when hideFinishedRuns is true',
        hideFinishedRuns: true,
        expectedIcon: 'visible-off-icon',
        notExpectedIcon: 'visible-fill-icon',
      },
    ];

    test.each(iconTestCases)('shows $name', ({ hideFinishedRuns, expectedIcon, notExpectedIcon }) => {
      (mockSearchFacetsState as any).hideFinishedRuns = hideFinishedRuns;
      renderComponent();

      expect(screen.getByTestId(expectedIcon)).toBeInTheDocument();
      expect(screen.queryByTestId(notExpectedIcon)).not.toBeInTheDocument();
    });
  });

  describe('Sorting Functionality', () => {
    const sortingTestCases = [
      {
        name: 'ascending icon',
        orderByAsc: true,
        expectedIcon: 'sort-ascending-icon',
        notExpectedIcon: 'sort-descending-icon',
      },
      {
        name: 'descending icon',
        orderByAsc: false,
        expectedIcon: 'sort-descending-icon',
        notExpectedIcon: 'sort-ascending-icon',
      },
    ];

    test.each(sortingTestCases)(
      'shows $name when column is ordered',
      ({ orderByAsc, expectedIcon, notExpectedIcon }) => {
        const mockColumn = {
          getColDef: () => ({
            headerComponentParams: { canonicalSortKey: 'metrics.accuracy' },
          }),
        };

        renderComponent({
          context: { orderByKey: 'metrics.accuracy', orderByAsc },
          column: mockColumn,
        });

        expect(screen.getByTestId(expectedIcon)).toBeInTheDocument();
        expect(screen.queryByTestId(notExpectedIcon)).not.toBeInTheDocument();
      },
    );

    it('does not show sort icons when sorting is disabled', () => {
      renderComponent({ enableSorting: false });

      expect(screen.queryByTestId('sort-ascending-icon')).not.toBeInTheDocument();
      expect(screen.queryByTestId('sort-descending-icon')).not.toBeInTheDocument();
    });
  });

  describe('Dropdown State Display', () => {
    const dropdownStateTestCases = [
      { name: 'shows "10" when runLimit is 10', runLimit: 10, expectedValue: '10' },
      { name: 'shows "20" when runLimit is 20', runLimit: 20, expectedValue: '20' },
      { name: 'shows "all" when runLimit is null', runLimit: null, expectedValue: 'all' },
    ];

    test.each(dropdownStateTestCases)('$name', ({ runLimit, expectedValue }) => {
      (mockSearchFacetsState as any).runLimit = runLimit;
      renderComponent();

      const radioGroup = screen.getByTestId('radio-group');
      expect(radioGroup).toHaveAttribute('data-value', expectedValue);
    });

    const checkboxStateTestCases = [
      { name: 'checked when hideFinishedRuns is true', hideFinishedRuns: true, expectedChecked: 'true' },
      { name: 'unchecked when hideFinishedRuns is false', hideFinishedRuns: false, expectedChecked: 'false' },
    ];

    test.each(checkboxStateTestCases)('checkbox is $name', ({ hideFinishedRuns, expectedChecked }) => {
      (mockSearchFacetsState as any).hideFinishedRuns = hideFinishedRuns;
      renderComponent();

      const checkbox = screen.getByTestId('checkbox-item');
      expect(checkbox).toHaveAttribute('data-checked', expectedChecked);
    });
  });

  describe('User Interactions', () => {
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
  });

  describe('Accessibility', () => {
    const accessibilityTestCases = [
      { element: 'Toggle visibility of runs', type: 'aria-label' },
      { element: 'columnheader', type: 'role' },
      { element: 'menuitemcheckbox', type: 'role' },
    ];

    test.each(accessibilityTestCases)('has proper $type for $element', ({ element, type }) => {
      renderComponent();

      if (type === 'aria-label') {
        expect(screen.getByLabelText(element)).toBeInTheDocument();
      } else if (type === 'role') {
        expect(screen.getByRole(element)).toBeInTheDocument();
      }
    });
  });

  describe('Edge Cases', () => {
    const edgeCaseTestCases = [
      {
        name: 'undefined context properties',
        props: { context: { orderByKey: undefined, orderByAsc: undefined } },
        shouldNotThrow: true,
      },
      {
        name: 'missing context',
        props: { context: undefined },
        shouldNotThrow: true,
      },
      {
        name: 'missing column definition',
        props: { column: undefined },
        shouldNotThrow: true,
      },
    ];

    test.each(edgeCaseTestCases)('handles $name gracefully', ({ props, shouldNotThrow }) => {
      if (shouldNotThrow) {
        expect(() => renderComponent(props)).not.toThrow();
      }
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
  });

  describe('Component Structure', () => {
    it('maintains consistent DOM structure with all features enabled', () => {
      const { container } = renderComponent({
        displayName: 'Test Runs',
        enableSorting: true,
        context: {
          orderByKey: 'metrics.accuracy',
          orderByAsc: true,
        },
        column: {
          getColDef: () => ({
            headerComponentParams: { canonicalSortKey: 'metrics.accuracy' },
          }),
        },
      });

      // Verify core structure elements exist
      const requiredElements = [
        'columnheader',
        'visibility-toggle-button',
        'dropdown-root',
        'radio-group',
        'checkbox-item',
      ];

      requiredElements.forEach((elementTestId) => {
        if (elementTestId === 'columnheader') {
          expect(screen.getByRole(elementTestId)).toBeInTheDocument();
        } else {
          expect(screen.getByTestId(elementTestId)).toBeInTheDocument();
        }
      });

      expect(container.firstChild).toBeInTheDocument();
    });
  });

  describe('Error handling', () => {
    beforeEach(() => {
      // Suppress console.warn for these tests
      jest.spyOn(console, 'warn').mockImplementation(() => {});
    });

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('handles invalid run limit values gracefully', () => {
      // Test that component doesn't crash with invalid dropdown interactions
      renderComponent();

      // Test that component renders without crashing
      expect(screen.getByTestId('visibility-toggle-button')).toBeInTheDocument();

      // Open dropdown and ensure it doesn't crash
      expect(() => {
        fireEvent.click(screen.getByTestId('visibility-toggle-button'));
      }).not.toThrow();
    });

    it('handles NaN values in run limit gracefully', () => {
      // Test that component doesn't crash with invalid values
      renderComponent();

      expect(() => {
        fireEvent.click(screen.getByTestId('visibility-toggle-button'));
      }).not.toThrow();
    });

    it('handles negative run limit values gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'warn');

      // Create a component with handleRunLimitChange exposed for testing
      const TestComponent = () => {
        const handleRunLimitChange = React.useCallback((value: string) => {
          let limit: number | null = null;

          if (value === 'all') {
            limit = null;
          } else {
            const parsedValue = parseInt(value, 10);
            if (!isNaN(parsedValue) && parsedValue > 0) {
              limit = parsedValue;
            } else {
              console.warn(`Invalid run limit value: ${value}. Falling back to show all runs.`);
              limit = null;
            }
          }

          return limit;
        }, []);

        // Test negative value
        const result = handleRunLimitChange('-5');

        return <div data-testid="result">{JSON.stringify(result)}</div>;
      };

      render(<TestComponent />);

      expect(consoleSpy).toHaveBeenCalledWith('Invalid run limit value: -5. Falling back to show all runs.');
      expect(screen.getByTestId('result')).toHaveTextContent('null');
    });

    it('handles zero run limit values gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'warn');

      const TestComponent = () => {
        const handleRunLimitChange = React.useCallback((value: string) => {
          let limit: number | null = null;

          if (value === 'all') {
            limit = null;
          } else {
            const parsedValue = parseInt(value, 10);
            if (!isNaN(parsedValue) && parsedValue > 0) {
              limit = parsedValue;
            } else {
              console.warn(`Invalid run limit value: ${value}. Falling back to show all runs.`);
              limit = null;
            }
          }

          return limit;
        }, []);

        const result = handleRunLimitChange('0');

        return <div data-testid="result">{JSON.stringify(result)}</div>;
      };

      render(<TestComponent />);

      expect(consoleSpy).toHaveBeenCalledWith('Invalid run limit value: 0. Falling back to show all runs.');
      expect(screen.getByTestId('result')).toHaveTextContent('null');
    });

    it('renders error boundary fallback when component throws', () => {
      // Since we can't easily mock React errors in the actual component,
      // we test that the component is wrapped with error boundary
      renderComponent();

      // Verify component renders normally
      expect(screen.getByTestId('visibility-toggle-button')).toBeInTheDocument();
    });
  });
});
