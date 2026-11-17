import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { fireEvent } from '@testing-library/react';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { ModelTraceExplorerChatMessage } from './ModelTraceExplorerChatMessage';
import * as chatRendererUtils from './ModelTraceExplorerChatRenderer.utils';

const mockCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({ useClipboard: () => ({ copy: mockCopy }) }));
jest.mock('../field-renderers/ModelTraceExplorerAttachmentRenderer', () => ({
  ModelTraceExplorerAttachmentRenderer: ({ attachmentId }: { attachmentId: string }) => (
    <span>attachment:{attachmentId}</span>
  ),
}));

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
    mockCopy.mockReset();
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

  it('keeps a large message preview in Markdown and bounds only its expansion', () => {
    mockViewport();
    const content = Array.from({ length: 3000 }, (_, index) => `**synthetic-${index}** with café 日本語 🙂`).join('\n');
    const { container } = render(<ModelTraceExplorerChatMessage message={{ role: 'assistant', content }} />, {
      wrapper: Providers,
    });
    const expectPreview = () => {
      expect(screen.queryByText(/Large content is displayed as plain text\./)).not.toBeInTheDocument();
      expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
      expect(container.querySelector('strong')).toHaveTextContent('synthetic-0');
    };
    expectPreview();
    fireEvent.click(screen.getByText('See more'));
    expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('synthetic-0');
    expect(container.textContent).not.toContain('synthetic-2999');
    expect(container.querySelectorAll('*').length).toBeLessThan(200);
    fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(mockCopy).toHaveBeenCalledWith(content);
    fireEvent.click(screen.getByText('See less'));
    expectPreview();
  });

  it.each(['image', 'attachment'])('preserves mixed large text and %s rendering', (kind) => {
    const uri =
      kind === 'image'
        ? 'https://example.invalid/image.png'
        : 'mlflow-attachment://synthetic-pdf?content_type=application%2Fpdf&trace_id=tr-synthetic';
    const content = `![synthetic](${uri}) **mixed café 🌳** ` + 'x'.repeat(100_001);
    const { container } = render(<ModelTraceExplorerChatMessage message={{ role: 'assistant', content }} />, {
      wrapper: Providers,
    });
    const expectMedia = () => {
      if (kind === 'image') {
        expect(screen.getByRole('img', { name: 'synthetic' })).toHaveAttribute('src', uri);
      } else {
        expect(screen.getByText('attachment:synthetic-pdf')).toBeInTheDocument();
      }
      expect(container.querySelector('strong')).toHaveTextContent('mixed café 🌳');
      expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
    };
    expectMedia();
    fireEvent.click(screen.getByText('See more'));
    expectMedia();
    expect(container.textContent).toContain('x'.repeat(100_001));
    fireEvent.click(screen.getByText('See less'));
    expectMedia();
  });

  it('bounds a large non-JSON tool response', () => {
    mockViewport();
    const content = 'synthetic tool response café 日本語 🙂\n'.repeat(4000);
    const { container } = render(
      <ModelTraceExplorerChatMessage message={{ role: 'tool', content, tool_call_id: 'synthetic-call' }} />,
      { wrapper: Providers },
    );
    expect(screen.queryByText(/Large content is displayed as plain text\./)).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('See more'));
    expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('synthetic tool response');
    expect(container.querySelectorAll('*').length).toBeLessThan(200);
    fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(mockCopy).toHaveBeenCalledWith(content);
    fireEvent.click(screen.getByText('See less'));
    expect(screen.queryByText(/Large content is displayed as plain text\./)).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
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
    fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(mockCopy).toHaveBeenCalledWith(reasoning);
    expect(container.querySelectorAll('*').length).toBeLessThan(200);
  });

  it('reuses reasoning classification across toggles and recomputes when reasoning changes', () => {
    mockViewport();
    const scan = jest.spyOn(chatRendererUtils, 'getDisplayLength');
    const reasoning = 'synthetic reasoning café 日本語 🙂\n'.repeat(4000);
    const message = { role: 'assistant' as const, content: 'small', reasoning };
    const { rerender } = render(<ModelTraceExplorerChatMessage message={message} />, { wrapper: Providers });
    const scans = () => scan.mock.calls.filter(([value]) => value === reasoning).length;
    expect(scans()).toBe(1);
    fireEvent.click(screen.getByRole('button', { name: 'Reasoning' }));
    fireEvent.click(screen.getByRole('button', { name: 'Reasoning' }));
    rerender(<ModelTraceExplorerChatMessage message={{ ...message }} />);
    expect(scans()).toBe(1);
    const replacement = reasoning + 'replacement';
    rerender(<ModelTraceExplorerChatMessage message={{ ...message, reasoning: replacement }} />);
    expect(scan.mock.calls.filter(([value]) => value === replacement)).toHaveLength(1);
  });

  it('preserves images in an expanded large reasoning section', () => {
    const reasoning = '![reasoning image](https://example.invalid/reasoning.png) ' + 'x'.repeat(100_001);
    render(<ModelTraceExplorerChatMessage message={{ role: 'assistant', content: 'small', reasoning }} />, {
      wrapper: Providers,
    });
    fireEvent.click(screen.getByRole('button', { name: 'Reasoning' }));
    expect(screen.getByRole('img', { name: 'reasoning image' })).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
  });

  it('keeps all large text available past the old one-megabyte cap', () => {
    mockViewport();
    const content = 'synthetic café 日本語 🙂\n'.repeat(50000);
    const { container } = render(<ModelTraceExplorerChatMessage message={{ role: 'assistant', content }} />, {
      wrapper: Providers,
    });
    fireEvent.click(screen.getByText('See more'));
    expect(screen.getByRole('region', { name: 'Trace text' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(mockCopy).toHaveBeenCalledWith(content);
    expect(container.querySelectorAll('*').length).toBeLessThan(200);
    expect(screen.queryByText(/Content too large to render/)).not.toBeInTheDocument();
  });

  it('retains the existing render cap for image-heavy messages', () => {
    const content = `![synthetic](data:image/png;base64,${'a'.repeat(1000100)})`;
    const { container } = render(<ModelTraceExplorerChatMessage message={{ role: 'assistant', content }} />, {
      wrapper: Providers,
    });
    expect(screen.getByText(/Content too large to render/)).toBeInTheDocument();
    expect(container.querySelector('pre')?.textContent).toHaveLength(10000);
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
  });
});
