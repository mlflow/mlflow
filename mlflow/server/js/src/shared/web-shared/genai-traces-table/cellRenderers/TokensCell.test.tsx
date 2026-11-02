import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TokenComponent } from './TokensCell';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('TokenComponent', () => {
  it('shows labeled input and output token counts on hover', async () => {
    render(<TokenComponent inputTokens={7} outputTokens={5} totalTokens={12} isComparing={false} />, {
      wrapper: Wrapper,
    });

    await userEvent.hover(screen.getByText('12'));

    expect(await screen.findByText('Total')).toBeInTheDocument();
    expect(screen.getByText('Input').parentElement?.nextElementSibling).toHaveTextContent('7');
    expect(screen.getByText('Output').parentElement?.nextElementSibling).toHaveTextContent('5');
  });

  it('shows a zero output token count with its label', async () => {
    render(<TokenComponent inputTokens={12} outputTokens={0} totalTokens={12} isComparing={false} />, {
      wrapper: Wrapper,
    });

    await userEvent.hover(screen.getByText('12'));

    expect(await screen.findByText('Total')).toBeInTheDocument();
    expect(screen.getByText('Output').parentElement?.nextElementSibling).toHaveTextContent('0');
  });

  it('shows a zero input token count with its label', async () => {
    render(<TokenComponent inputTokens={0} outputTokens={12} totalTokens={12} isComparing={false} />, {
      wrapper: Wrapper,
    });

    await userEvent.hover(screen.getByText('12'));

    expect(await screen.findByText('Total')).toBeInTheDocument();
    expect(screen.getByText('Input').parentElement?.nextElementSibling).toHaveTextContent('0');
  });

  it('omits input and output rows when the counts are missing', async () => {
    render(<TokenComponent totalTokens={12} isComparing={false} />, { wrapper: Wrapper });

    await userEvent.hover(screen.getByText('12'));

    expect(await screen.findByText('Total')).toBeInTheDocument();
    expect(screen.queryByText('Input')).not.toBeInTheDocument();
    expect(screen.queryByText('Output')).not.toBeInTheDocument();
  });
});
