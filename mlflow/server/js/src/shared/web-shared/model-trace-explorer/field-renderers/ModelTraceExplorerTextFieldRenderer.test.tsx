import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerTextFieldRenderer } from './ModelTraceExplorerTextFieldRenderer';

const renderWithProviders = (ui: React.ReactElement) =>
  render(
    <DesignSystemProvider>
      <IntlProvider locale="en">{ui}</IntlProvider>
    </DesignSystemProvider>,
  );

const COPY_ACTION_SELECTOR = '[data-component-id="shared.model-trace-explorer.copy-text-field"]';
// longer than STRING_TRUNCATION_LIMIT (400), so the preview is truncated behind "See more"
const longValue = 'a'.repeat(500);

describe('ModelTraceExplorerTextFieldRenderer', () => {
  beforeEach(() => {
    Object.defineProperty(global.navigator, 'clipboard', {
      value: { writeText: jest.fn(async () => {}) },
      writable: true,
    });
  });

  it('copies the full value while the preview is truncated', async () => {
    const { container } = renderWithProviders(<ModelTraceExplorerTextFieldRenderer title="prompt" value={longValue} />);

    expect(screen.getByText('See more')).toBeInTheDocument();

    await userEvent.click(container.querySelector(COPY_ACTION_SELECTOR) as Element);

    expect(global.navigator.clipboard.writeText).toHaveBeenCalledWith(longValue);
  });

  it('copies the full value when the field is expanded', async () => {
    const { container } = renderWithProviders(<ModelTraceExplorerTextFieldRenderer title="prompt" value={longValue} />);

    await userEvent.click(screen.getByText('See more'));
    expect(screen.getByText('See less')).toBeInTheDocument();

    await userEvent.click(container.querySelector(COPY_ACTION_SELECTOR) as Element);

    expect(global.navigator.clipboard.writeText).toHaveBeenCalledWith(longValue);
  });

  it('renders the copy action for short values that are not expandable', () => {
    const { container } = renderWithProviders(<ModelTraceExplorerTextFieldRenderer title="prompt" value="short" />);

    expect(screen.queryByText('See more')).not.toBeInTheDocument();
    expect(container.querySelector(COPY_ACTION_SELECTOR)).not.toBeNull();
  });
});
