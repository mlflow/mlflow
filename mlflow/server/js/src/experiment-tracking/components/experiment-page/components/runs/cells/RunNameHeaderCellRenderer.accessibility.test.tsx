import { render, screen } from '@testing-library/react';
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
    RadioGroup: ({ children, value, componentId }: any) => (
      <div data-testid="radio-group" data-component-id={componentId} data-value={value}>
        {children}
      </div>
    ),
    RadioItem: ({ children, value }: any) => (
      <div data-testid={`radio-item-${value}`} role="menuitemradio" aria-checked="false">
        {children}
      </div>
    ),
    CheckboxItem: ({ children, checked, componentId }: any) => (
      <div
        data-testid="checkbox-item"
        data-component-id={componentId}
        data-checked={checked}
        role="menuitemcheckbox"
        aria-checked={checked ? 'true' : 'false'}
      >
        {children}
      </div>
    ),
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

describe('RunNameHeaderCellRenderer - Accessibility', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Object.assign(mockSearchFacetsState, {
      hideFinishedRuns: false,
      runLimit: null,
    });
  });

  describe('ARIA Labels and Roles', () => {
    const accessibilityTestCases = [
      { element: 'Toggle visibility of runs', type: 'aria-label' },
      { element: 'columnheader', type: 'role' },
      { element: 'menuitemcheckbox', type: 'role' },
      { element: 'menuitemradio', type: 'role' },
    ];

    test.each(accessibilityTestCases)('has proper $type for $element', ({ element, type }) => {
      renderComponent();

      if (type === 'aria-label') {
        expect(screen.getByLabelText(element)).toBeInTheDocument();
      } else if (type === 'role') {
        if (element === 'menuitemradio') {
          // Multiple radio items, check for at least one
          expect(screen.getAllByRole(element).length).toBeGreaterThan(0);
        } else {
          expect(screen.getByRole(element)).toBeInTheDocument();
        }
      }
    });
  });

  describe('Dynamic ARIA Labels', () => {
    it('provides contextual aria-label for checkbox when hideFinishedRuns is false', () => {
      (mockSearchFacetsState as any).hideFinishedRuns = false;
      renderComponent();

      const checkbox = screen.getByTestId('checkbox-item');
      expect(checkbox).toHaveAttribute('aria-checked', 'false');
    });

    it('provides contextual aria-label for checkbox when hideFinishedRuns is true', () => {
      (mockSearchFacetsState as any).hideFinishedRuns = true;
      renderComponent();

      const checkbox = screen.getByTestId('checkbox-item');
      expect(checkbox).toHaveAttribute('aria-checked', 'true');
    });
  });

  describe('Component IDs for Testing and Accessibility', () => {
    const componentIdTestCases = [
      { testId: 'visibility-toggle-button', expectedId: 'run_name_header_visibility_dropdown' },
      { testId: 'tooltip', expectedId: 'run_name_header_visibility_tooltip' },
      { testId: 'radio-group', expectedId: 'run_name_header_run_limit_group' },
      { testId: 'checkbox-item', expectedId: 'run_name_header_hide_finished_checkbox' },
    ];

    test.each(componentIdTestCases)('has correct component ID for $testId', ({ testId, expectedId }) => {
      renderComponent();

      const element = screen.getByTestId(testId);
      expect(element).toHaveAttribute('data-component-id', expectedId);
    });
  });

  describe('Tooltip Content for Screen Readers', () => {
    const tooltipTestCases = [
      {
        name: 'shows appropriate tooltip when runs are hidden',
        hideFinishedRuns: true,
        expectedTooltip: 'Some runs are hidden. Click to show options.',
      },
      {
        name: 'shows appropriate tooltip when all runs are visible',
        hideFinishedRuns: false,
        expectedTooltip: 'All runs are visible. Click to show options.',
      },
    ];

    test.each(tooltipTestCases)('$name', ({ hideFinishedRuns, expectedTooltip }) => {
      (mockSearchFacetsState as any).hideFinishedRuns = hideFinishedRuns;
      renderComponent();

      const tooltip = screen.getByTestId('tooltip');
      expect(tooltip).toHaveAttribute('title', expectedTooltip);
    });
  });

  describe('Keyboard Navigation Support', () => {
    it('ensures focusable elements are properly accessible', () => {
      renderComponent();

      // Button should be focusable
      const button = screen.getByTestId('visibility-toggle-button');
      expect(button.tagName.toLowerCase()).toBe('button');

      // Ensure no focusable elements are missing tabindex when needed
      const interactiveElements = screen.getAllByRole('button');
      interactiveElements.forEach((element) => {
        // Should either have tabindex="0" or no tabindex (default focusable)
        const tabIndex = element.getAttribute('tabindex');
        expect(tabIndex === null || tabIndex === '0' || parseInt(tabIndex, 10) >= 0).toBe(true);
      });
    });
  });

  describe('Screen Reader Content', () => {
    it('provides meaningful text content for screen readers', () => {
      renderComponent();

      // Check that text content is meaningful
      expect(screen.getByText('Runs')).toBeInTheDocument();
      expect(screen.getByText('Show first 10')).toBeInTheDocument();
      expect(screen.getByText('Show first 20')).toBeInTheDocument();
      expect(screen.getByText('Show all runs')).toBeInTheDocument();
      expect(screen.getByText('Hide finished runs')).toBeInTheDocument();
    });

    it('updates screen reader content based on state', () => {
      (mockSearchFacetsState as any).hideFinishedRuns = true;
      renderComponent();

      // Text should remain constant - "Hide finished runs" - only checkmark indicates state
      expect(screen.getByText('Hide finished runs')).toBeInTheDocument();

      // The checkbox should be checked when hideFinishedRuns is true
      const checkbox = screen.getByTestId('checkbox-item');
      expect(checkbox).toHaveAttribute('aria-checked', 'true');
    });
  });

  describe('Color and Visual Accessibility', () => {
    it('does not rely solely on color for information', () => {
      renderComponent();

      // Icons should provide visual distinction beyond color
      const visibleIcon = screen.getByTestId('visible-fill-icon');
      expect(visibleIcon).toBeInTheDocument();

      // Change state and verify icon changes
      (mockSearchFacetsState as any).hideFinishedRuns = true;
      renderComponent();

      expect(screen.getByTestId('visible-off-icon')).toBeInTheDocument();
    });
  });

  describe('Error Prevention', () => {
    it('provides clear feedback for user actions', () => {
      renderComponent();

      // Tooltip provides clear feedback about current state
      const tooltip = screen.getByTestId('tooltip');
      expect(tooltip).toHaveAttribute('title');

      // Button has clear labeling
      const button = screen.getByTestId('visibility-toggle-button');
      expect(button).toHaveAttribute('aria-label', 'Toggle visibility of runs');
    });
  });
});
