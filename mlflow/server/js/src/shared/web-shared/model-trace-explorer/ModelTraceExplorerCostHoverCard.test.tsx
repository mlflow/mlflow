import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerCostHoverCard, isTraceCostType, type TraceCost } from './ModelTraceExplorerCostHoverCard';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('isTraceCostType', () => {
  it('should return true for valid TraceCost objects', () => {
    const validCost: TraceCost = {
      input_cost: 0.001,
      output_cost: 0.002,
      total_cost: 0.003,
    };
    expect(isTraceCostType(validCost)).toBe(true);
  });

  it('should return true for TraceCost with zero values', () => {
    const zeroCost = {
      input_cost: 0,
      output_cost: 0,
      total_cost: 0,
    };
    expect(isTraceCostType(zeroCost)).toBe(true);
  });

  it('should return false for undefined', () => {
    expect(isTraceCostType(undefined)).toBe(false);
  });

  it('should return false for null', () => {
    expect(isTraceCostType(null)).toBe(false);
  });

  it('should return false for empty object', () => {
    expect(isTraceCostType({})).toBe(false);
  });

  it('should return false for object missing input_cost', () => {
    expect(isTraceCostType({ output_cost: 0.002, total_cost: 0.003 })).toBe(false);
  });

  it('should return false for object missing output_cost', () => {
    expect(isTraceCostType({ input_cost: 0.001, total_cost: 0.003 })).toBe(false);
  });

  it('should return false for object missing total_cost', () => {
    expect(isTraceCostType({ input_cost: 0.001, output_cost: 0.002 })).toBe(false);
  });

  it('should return false for non-object values', () => {
    expect(isTraceCostType('string')).toBe(false);
    expect(isTraceCostType(123)).toBe(false);
    expect(isTraceCostType([])).toBe(false);
  });
});

describe('ModelTraceExplorerCostHoverCard', () => {
  it('renders cost label and total cost', () => {
    const cost: TraceCost = {
      input_cost: 0.001,
      output_cost: 0.002,
      total_cost: 0.003,
    };
    render(<ModelTraceExplorerCostHoverCard cost={cost} />, { wrapper: Wrapper });

    expect(screen.getByText('Cost')).toBeInTheDocument();
    expect(screen.getByText('$0.003')).toBeInTheDocument();
  });

  it('renders formatted cost with appropriate precision', () => {
    const cost: TraceCost = {
      input_cost: 0.000022,
      output_cost: 0.000028,
      total_cost: 0.00005,
    };
    render(<ModelTraceExplorerCostHoverCard cost={cost} />, { wrapper: Wrapper });

    expect(screen.getByText('$0.00005')).toBeInTheDocument();
  });

  it('shows cost breakdown on hover', async () => {
    const cost: TraceCost = {
      input_cost: 0.001234,
      output_cost: 0.002345,
      total_cost: 0.003579,
    };
    render(<ModelTraceExplorerCostHoverCard cost={cost} />, { wrapper: Wrapper });

    // Hover over the cost tag to trigger the hover card
    const costTrigger = screen.getByText('$0.003579');
    await userEvent.hover(costTrigger);

    // Check that the breakdown is shown
    expect(await screen.findByText('Cost breakdown')).toBeInTheDocument();
    expect(screen.getByText('Input cost')).toBeInTheDocument();
    expect(screen.getByText('Output cost')).toBeInTheDocument();
    expect(screen.getByText('Total')).toBeInTheDocument();
    expect(screen.getByText('$0.001234')).toBeInTheDocument();
    expect(screen.getByText('$0.002345')).toBeInTheDocument();
  });

  it('handles zero costs correctly', async () => {
    const cost: TraceCost = {
      input_cost: 0,
      output_cost: 0,
      total_cost: 0,
    };
    render(<ModelTraceExplorerCostHoverCard cost={cost} />, { wrapper: Wrapper });

    expect(screen.getByText('$0.00')).toBeInTheDocument();

    // Hover to see breakdown
    const costTrigger = screen.getByText('$0.00');
    await userEvent.hover(costTrigger);

    expect(await screen.findByText('Cost breakdown')).toBeInTheDocument();
  });

  it('handles large costs correctly', () => {
    const cost: TraceCost = {
      input_cost: 1.5,
      output_cost: 2.5,
      total_cost: 4.0,
    };
    render(<ModelTraceExplorerCostHoverCard cost={cost} />, { wrapper: Wrapper });

    expect(screen.getByText('$4.00')).toBeInTheDocument();
  });
});
