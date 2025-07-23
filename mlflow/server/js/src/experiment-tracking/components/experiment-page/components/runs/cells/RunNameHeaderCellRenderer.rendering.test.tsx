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

describe('RunNameHeaderCellRenderer - Rendering', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Object.assign(mockSearchFacetsState, {
      hideFinishedRuns: false,
      runLimit: null,
    });
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
});
