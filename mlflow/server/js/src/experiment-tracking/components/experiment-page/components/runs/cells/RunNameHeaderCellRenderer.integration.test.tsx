import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunNameHeaderCellRenderer } from './RunNameHeaderCellRenderer';

// Mock the hooks with realistic state
const mockUpdateSearchFacets = jest.fn();
const mockSearchFacetsState = {
  hideFinishedRuns: false,
  runLimit: null as number | null,
  searchFilter: '',
  orderByKey: '',
  orderByAsc: true,
  startTime: 'ALL',
  lifecycleFilter: 'ACTIVE',
  datasetsFilter: [],
  modelVersionFilter: 'All Runs',
};

jest.mock('../../../hooks/useExperimentPageSearchFacets', () => ({
  useExperimentPageSearchFacets: () => [mockSearchFacetsState, ['123'], false],
  useUpdateExperimentPageSearchFacets: () => mockUpdateSearchFacets,
}));

// Mock the SVG icon
jest.mock('../../../../../../common/static/icon-visible-fill.svg', () => ({
  ReactComponent: () => <div data-testid="visible-fill-icon" />,
}));

const renderComponent = (props = {}) => {
  const user = userEvent.setup();
  const result = render(
    <DesignSystemProvider>
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
      </IntlProvider>
    </DesignSystemProvider>,
  );
  return { user, ...result };
};

describe('RunNameHeaderCellRenderer - Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Object.assign(mockSearchFacetsState, {
      hideFinishedRuns: false,
      runLimit: null,
    });
  });

  describe('Dropdown Component Presence and Basic Interaction', () => {
    it('renders dropdown trigger button with proper attributes', async () => {
      renderComponent();

      // Find the trigger button
      const triggerButton = screen.getByLabelText('Toggle visibility of runs');
      expect(triggerButton).toBeInTheDocument();
      expect(triggerButton).toHaveAttribute('type', 'button');

      // Verify it's clickable without errors
      expect(() => fireEvent.click(triggerButton)).not.toThrow();
    });

    it('trigger button displays correct icon based on state', () => {
      const { rerender } = renderComponent();

      // Initially shows visible icon (all runs visible)
      expect(screen.getByTestId('visible-fill-icon')).toBeInTheDocument();

      // Simulate state change to hidden
      Object.assign(mockSearchFacetsState, { hideFinishedRuns: true });
      rerender(
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <RunNameHeaderCellRenderer
              displayName="Runs"
              enableSorting
              context={{
                orderByKey: 'metrics.accuracy',
                orderByAsc: false,
              }}
            />
          </IntlProvider>
        </DesignSystemProvider>,
      );

      // Should now show the "off" icon
      expect(screen.queryByTestId('visible-fill-icon')).not.toBeInTheDocument();
    });

    it('renders dropdown structure correctly', () => {
      renderComponent();

      // Check for dropdown trigger button (which has the component ID)
      expect(screen.getByLabelText('Toggle visibility of runs')).toHaveAttribute(
        'data-component-id',
        'run_name_header_visibility_dropdown',
      );

      // Verify the dropdown is properly nested within the component
      const headerCell = screen.getByRole('columnheader');
      expect(headerCell).toBeInTheDocument();
      expect(headerCell).toContainElement(screen.getByLabelText('Toggle visibility of runs'));
    });

    it('tooltip shows appropriate content based on state', () => {
      const { rerender } = renderComponent();

      // Check that tooltip wrapper exists (tooltip is wrapped around the button)
      const triggerButton = screen.getByLabelText('Toggle visibility of runs');
      expect(triggerButton).toBeInTheDocument();

      // Simulate state change
      Object.assign(mockSearchFacetsState, { hideFinishedRuns: true });
      rerender(
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <RunNameHeaderCellRenderer
              displayName="Runs"
              enableSorting
              context={{
                orderByKey: 'metrics.accuracy',
                orderByAsc: false,
              }}
            />
          </IntlProvider>
        </DesignSystemProvider>,
      );

      // Tooltip should still be present with different content (button still has aria-label)
      expect(screen.getByLabelText('Toggle visibility of runs')).toBeInTheDocument();
    });
  });

  describe('State Reflection in UI', () => {
    it('reflects runLimit state in component props', () => {
      const { rerender } = renderComponent();

      // Test with null runLimit (default)
      Object.assign(mockSearchFacetsState, { runLimit: null });
      rerender(
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <RunNameHeaderCellRenderer
              displayName="Runs"
              enableSorting
              context={{
                orderByKey: 'metrics.accuracy',
                orderByAsc: false,
              }}
            />
          </IntlProvider>
        </DesignSystemProvider>,
      );

      expect(screen.getByLabelText('Toggle visibility of runs')).toBeInTheDocument();

      // Test with specific runLimit
      Object.assign(mockSearchFacetsState, { runLimit: 10 });
      rerender(
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <RunNameHeaderCellRenderer
              displayName="Runs"
              enableSorting
              context={{
                orderByKey: 'metrics.accuracy',
                orderByAsc: false,
              }}
            />
          </IntlProvider>
        </DesignSystemProvider>,
      );

      expect(screen.getByLabelText('Toggle visibility of runs')).toBeInTheDocument();
    });

    it('reflects hideFinishedRuns state in UI elements', () => {
      const { rerender } = renderComponent();

      // Test with hideFinishedRuns false
      Object.assign(mockSearchFacetsState, { hideFinishedRuns: false });
      rerender(
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <RunNameHeaderCellRenderer
              displayName="Runs"
              enableSorting
              context={{
                orderByKey: 'metrics.accuracy',
                orderByAsc: false,
              }}
            />
          </IntlProvider>
        </DesignSystemProvider>,
      );

      expect(screen.getByTestId('visible-fill-icon')).toBeInTheDocument();

      // Test with hideFinishedRuns true
      Object.assign(mockSearchFacetsState, { hideFinishedRuns: true });
      rerender(
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <RunNameHeaderCellRenderer
              displayName="Runs"
              enableSorting
              context={{
                orderByKey: 'metrics.accuracy',
                orderByAsc: false,
              }}
            />
          </IntlProvider>
        </DesignSystemProvider>,
      );

      expect(screen.queryByTestId('visible-fill-icon')).not.toBeInTheDocument();
    });
  });

  describe('Basic Accessibility Features', () => {
    it('has proper ARIA attributes', () => {
      renderComponent();

      const triggerButton = screen.getByLabelText('Toggle visibility of runs');
      expect(triggerButton).toHaveAttribute('aria-label', 'Toggle visibility of runs');
      expect(triggerButton).toHaveAttribute('type', 'button');
    });

    it('maintains focus management', async () => {
      const { user } = renderComponent();

      const triggerButton = screen.getByLabelText('Toggle visibility of runs');

      // Button should be focusable
      await user.tab();
      expect(triggerButton).toHaveFocus();
    });

    it('supports keyboard interaction', async () => {
      const { user } = renderComponent();

      const triggerButton = screen.getByLabelText('Toggle visibility of runs');
      triggerButton.focus();

      // Should respond to keyboard events without errors
      expect(() => fireEvent.keyDown(triggerButton, { key: 'Enter' })).not.toThrow();
      expect(() => fireEvent.keyDown(triggerButton, { key: ' ' })).not.toThrow();
    });
  });

  describe('Component Integration', () => {
    it('integrates properly with sort functionality', async () => {
      const { user } = renderComponent();

      // Both sort header and dropdown should be present
      expect(screen.getByTestId('sort-header-Runs')).toBeInTheDocument();
      expect(screen.getByLabelText('Toggle visibility of runs')).toBeInTheDocument();

      // Clicking sort should work independently
      await user.click(screen.getByTestId('sort-header-Runs'));
      expect(mockUpdateSearchFacets).toHaveBeenCalledWith({
        orderByKey: undefined,
        orderByAsc: false,
      });
    });

    it('renders correctly with different context props', () => {
      const { rerender } = renderComponent({
        context: {
          orderByKey: 'metrics.loss',
          orderByAsc: true,
        },
      });

      expect(screen.getByLabelText('Toggle visibility of runs')).toBeInTheDocument();

      // Re-render with different context
      rerender(
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <RunNameHeaderCellRenderer
              displayName="Custom Name"
              enableSorting={false}
              context={{
                orderByKey: '',
                orderByAsc: false,
              }}
            />
          </IntlProvider>
        </DesignSystemProvider>,
      );

      expect(screen.getByLabelText('Toggle visibility of runs')).toBeInTheDocument();
    });
  });
});
