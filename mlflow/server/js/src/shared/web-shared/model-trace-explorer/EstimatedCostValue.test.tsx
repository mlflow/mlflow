import { describe, it, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { EstimatedCostValue } from './EstimatedCostValue';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('EstimatedCostValue', () => {
  it('renders the ≈-prefixed formatted cost', () => {
    render(<EstimatedCostValue cost={1.0} />, { wrapper: Wrapper });
    expect(screen.getByText('≈$1.00')).toBeVisible();
  });

  it('carries an accessible label announcing the value as estimated cost', () => {
    render(<EstimatedCostValue cost={1.0} />, { wrapper: Wrapper });
    expect(screen.getByLabelText('Estimated cost: $1.00')).toBeVisible();
  });

  it('shows the estimation-source tooltip on hover', async () => {
    render(<EstimatedCostValue cost={1.0} />, { wrapper: Wrapper });

    await userEvent.hover(screen.getByText('≈$1.00'));

    const tooltip = await screen.findByRole('tooltip');
    expect(tooltip).toHaveTextContent('Cost is estimated from token usage and public model pricing.');
  });

  it('respects an optional maximumFractionDigits override', () => {
    render(<EstimatedCostValue cost={0.001234} maximumFractionDigits={4} />, { wrapper: Wrapper });
    expect(screen.getByText('≈$0.0012')).toBeVisible();
  });
});
