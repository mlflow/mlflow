import { describe, expect, it } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { normalizeConversation } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerChatMessage } from '../right-pane/ModelTraceExplorerChatMessage';
import { ModelTraceExplorerChatMessage as ModelTraceExplorerChatMessageV2 } from '../v2/right-pane/ModelTraceExplorerChatMessage';

describe.each([
  { name: 'legacy', Component: ModelTraceExplorerChatMessage },
  { name: 'v2', Component: ModelTraceExplorerChatMessageV2 },
])('OTEL GenAI chat rendering ($name)', ({ Component }) => {
  it('renders mixed text, reasoning, media, and generic parts with the existing renderers', () => {
    const message = normalizeConversation([
      {
        role: 'assistant',
        parts: [
          { type: 'reasoning', content: 'Use a weather lookup.' },
          { type: 'text', content: 'The weather is sunny.' },
          { type: 'blob', modality: 'image', mime_type: 'image/png', content: 'aW1hZ2U=' },
          { type: 'blob', modality: 'audio', mime_type: 'audio/wav', content: 'YXVkaW8=' },
          { type: 'file', modality: 'image', file_id: 'file-123' },
          { type: 'provider_extension', content: { detail: 'Preserve this detail.' } },
        ],
      },
    ])![0];

    const { container } = render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <Component message={message} />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(screen.getByText('The weather is sunny.')).toBeInTheDocument();
    expect(screen.queryByText('Use a weather lookup.')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reasoning' }));
    expect(screen.getByText('Use a weather lookup.')).toBeInTheDocument();
    expect(container.querySelector('img')).toHaveAttribute('src', 'data:image/png;base64,aW1hZ2U=');
    expect(container.querySelector('audio')).toHaveAttribute('src', 'data:audio/wav;base64,YXVkaW8=');
    expect(container.textContent).toContain('file-123');
    expect(container.textContent).toContain('Preserve this detail.');
    expect(container.querySelectorAll('pre code')).toHaveLength(2);
  });

  it('keeps unsupported URI schemes inspectable without creating image elements', () => {
    const message = normalizeConversation([
      // eslint-disable-next-line no-script-url
      { role: 'user', parts: [{ type: 'uri', modality: 'image', uri: 'javascript:alert(1)' }] },
    ])![0];

    const { container } = render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <Component message={message} />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    // eslint-disable-next-line no-script-url
    expect(container.textContent).toContain('javascript:alert(1)');
    expect(container.querySelector('img')).not.toBeInTheDocument();
  });
});
