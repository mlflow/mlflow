import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { fireEvent, render } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerTextFieldRenderer } from './ModelTraceExplorerTextFieldRenderer';

const mockClipboardCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({
  useClipboard: () => ({ copy: mockClipboardCopy }),
}));

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

beforeEach(() => {
  mockClipboardCopy.mockClear();
});

describe('ModelTraceExplorerTextFieldRenderer', () => {
  it('copies the full value when the displayed preview is truncated', () => {
    const value = `# Heading\n\n${'long markdown content '.repeat(30)}`;
    const { container } = render(<ModelTraceExplorerTextFieldRenderer title="Output" value={value} />, {
      wrapper: Wrapper,
    });

    const copyButton = container.querySelector('[data-component-id="shared.model-trace-explorer.text-field-copy"]');
    expect(copyButton).not.toBeNull();
    if (!copyButton) return;

    fireEvent.click(copyButton);

    expect(mockClipboardCopy).toHaveBeenCalledWith(value);
  });
});
