import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { SnippetCopyAction } from './SnippetCopyAction';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('deferred snippet copy text', () => {
  beforeEach(() => {
    Object.defineProperty(global.navigator, 'clipboard', {
      value: { writeText: jest.fn(async () => {}) },
      writable: true,
    });
  });

  it('evaluates only the current callback when the user clicks', async () => {
    const first = jest.fn(() => 'previous');
    const current = jest.fn(() => 'café 日本語 🙂');
    const { rerender } = render(<SnippetCopyAction componentId="test.copy" copyText={first} />, { wrapper: Wrapper });
    rerender(<SnippetCopyAction componentId="test.copy" copyText={current} />);
    expect(first).not.toHaveBeenCalled();
    expect(current).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button'));
    expect(first).not.toHaveBeenCalled();
    expect(current).toHaveBeenCalledTimes(1);
    expect(global.navigator.clipboard.writeText).toHaveBeenCalledWith('café 日本語 🙂');
  });
});
