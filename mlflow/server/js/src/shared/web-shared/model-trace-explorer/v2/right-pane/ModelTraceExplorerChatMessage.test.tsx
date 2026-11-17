import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { fireEvent } from '@testing-library/react';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { ModelTraceExplorerChatMessage } from './ModelTraceExplorerChatMessage';

const Providers = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const mockViewport = () =>
  jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const height = this.hasAttribute('data-index') ? 20 : 400;
    return { x: 0, y: 0, top: 0, left: 0, width: 1000, height, right: 1000, bottom: height, toJSON: () => ({}) };
  });

describe('large chat text', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('preserves small Markdown/images and reuses unchanged content normalization', () => {
    const message: ModelTraceChatMessage = {
      role: 'assistant',
      content: '**small** ![synthetic](https://example.invalid/image.png)',
    };
    const parse = jest.spyOn(JSON, 'parse');
    const { container, rerender } = render(<ModelTraceExplorerChatMessage message={message} />, { wrapper: Providers });
    expect(container.querySelector('strong')).toHaveTextContent('small');
    expect(screen.getByRole('img', { name: 'synthetic' })).toBeInTheDocument();
    const attempts = () => parse.mock.calls.filter(([text]) => text === message.content).length;
    expect(attempts()).toBe(1);
    rerender(<ModelTraceExplorerChatMessage message={{ ...message }} />);
    expect(attempts()).toBe(1);
  });

  it('keeps large messages as bounded plain text before and after expansion', () => {
    mockViewport();
    const content = Array.from({ length: 3000 }, (_, index) => `**synthetic-${index}** with café 日本語 🙂`).join('\n');
    const { container } = render(<ModelTraceExplorerChatMessage message={{ role: 'assistant', content }} />, {
      wrapper: Providers,
    });
    expect(screen.getByText('Large content is displayed as plain text.')).toBeInTheDocument();
    expect(container.querySelector('strong')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('See more'));
    expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('synthetic-0');
    expect(container.textContent).not.toContain('synthetic-2999');
    expect(container.querySelectorAll('*').length).toBeLessThan(200);
  });

  it('bounds a large non-JSON tool response', () => {
    mockViewport();
    const content = 'synthetic tool response café 日本語 🙂\n'.repeat(4000);
    const { container } = render(
      <ModelTraceExplorerChatMessage message={{ role: 'tool', content, tool_call_id: 'synthetic-call' }} />,
      { wrapper: Providers },
    );
    fireEvent.click(screen.getByText('See more'));
    expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('synthetic tool response');
    expect(container.querySelectorAll('*').length).toBeLessThan(200);
  });

  it('bounds an expanded large reasoning section', () => {
    mockViewport();
    const reasoning = 'synthetic reasoning café 日本語 🙂\n'.repeat(4000);
    const { container } = render(
      <ModelTraceExplorerChatMessage message={{ role: 'assistant', content: 'small', reasoning }} />,
      { wrapper: Providers },
    );
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reasoning' }));
    expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('synthetic reasoning');
    expect(container.querySelectorAll('*').length).toBeLessThan(200);
  });
});
