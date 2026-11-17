import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { fireEvent } from '@testing-library/react';

import { ModelTraceExplorerTextFieldRenderer } from './ModelTraceExplorerTextFieldRenderer';

const Providers = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('large trace text rendering', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('keeps Markdown for small values', () => {
    const { container } = render(<ModelTraceExplorerTextFieldRenderer title="payload" value="**small**" />, {
      wrapper: Providers,
    });
    expect(container.querySelector('strong')).toHaveTextContent('small');
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
  });

  it('bounds the DOM for a large expanded value', () => {
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      const height = this.hasAttribute('data-index') ? 20 : 400;
      return { x: 0, y: 0, top: 0, left: 0, width: 1000, height, right: 1000, bottom: height, toJSON: () => ({}) };
    });
    const value = Array.from({ length: 3000 }, (_, i) => `- **synthetic-${i}** with \`code\` and café 日本語 🙂`).join(
      '\n',
    );
    expect(value.length).toBeGreaterThan(100000);
    const { container } = render(<ModelTraceExplorerTextFieldRenderer title="payload" value={value} />, {
      wrapper: Providers,
    });
    fireEvent.click(screen.getByText('See more'));
    const expandedNodes = container.querySelectorAll('*').length;
    expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('synthetic-0');
    expect(container.textContent).not.toContain('synthetic-2999');
    expect(expandedNodes).toBeLessThan(200);
    expect(screen.getByText(/Large content is displayed as plain text\./)).toBeInTheDocument();
  });
});
