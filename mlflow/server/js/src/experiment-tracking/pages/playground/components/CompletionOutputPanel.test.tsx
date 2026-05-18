import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { CompletionOutputPanel } from './CompletionOutputPanel';

const renderPanel = (props: React.ComponentProps<typeof CompletionOutputPanel>) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <CompletionOutputPanel {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('CompletionOutputPanel', () => {
  it('shows the empty state when there is no response, error, or loading', () => {
    renderPanel({ isLoading: false });
    expect(screen.getByText('Submit a message to see the response here.')).toBeInTheDocument();
  });

  it('hides the empty state, completion, and error UI while loading', () => {
    renderPanel({ isLoading: true });
    expect(screen.queryByText('Submit a message to see the response here.')).not.toBeInTheDocument();
    expect(screen.queryByText(/Chat completion failed/i)).not.toBeInTheDocument();
  });

  it('renders the completion text when a response is provided', () => {
    renderPanel({
      isLoading: false,
      response: {
        choices: [{ index: 0, message: { role: 'assistant', content: 'Hello back!' }, finish_reason: 'stop' }],
      },
    });
    expect(screen.getByText('Hello back!')).toBeInTheDocument();
    expect(screen.queryByText(/Chat completion failed/i)).not.toBeInTheDocument();
  });

  it('renders the error title with the upstream message when error has no status', () => {
    renderPanel({ isLoading: false, error: new Error('Something broke') });
    expect(screen.getByText('Chat completion failed')).toBeInTheDocument();
    expect(screen.getByText('Something broke')).toBeInTheDocument();
    expect(screen.queryByText(/^HTTP /)).not.toBeInTheDocument();
  });

  it('prefixes the description with the HTTP status when error carries one', () => {
    const error = Object.assign(new Error('Internal server error'), { status: 500 });
    renderPanel({ isLoading: false, error });
    expect(screen.getByText('Chat completion failed')).toBeInTheDocument();
    expect(screen.getByText('HTTP 500 — Internal server error')).toBeInTheDocument();
  });

  it('preserves newlines in multi-line provider error messages', () => {
    const error = Object.assign(new Error('line1\nline2'), { status: 400 });
    const { container } = renderPanel({ isLoading: false, error });
    const pre = container.querySelector('pre');
    expect(pre).not.toBeNull();
    expect(pre!.textContent).toBe('HTTP 400 — line1\nline2');
  });
});
