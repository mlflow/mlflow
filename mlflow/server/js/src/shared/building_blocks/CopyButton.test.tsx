import { describe, beforeEach, jest, afterEach, it, expect } from '@jest/globals';
import React from 'react';
import { CopyButton } from './CopyButton';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

describe('CopyButton', () => {
  const originalClipboard = { ...global.navigator.clipboard };
  beforeEach(() => {
    const mockClipboard = {
      writeText: jest.fn(),
    };
    Object.defineProperty(global.navigator, 'clipboard', {
      value: mockClipboard,
      writable: true,
    });
  });

  afterEach(() => {
    jest.resetAllMocks();
    Object.defineProperty(global.navigator, 'clipboard', {
      value: originalClipboard,
      writable: false,
    });
  });

  it('should render with minimal props without exploding', async () => {
    renderWithIntl(
      <DesignSystemProvider>
        <CopyButton copyText="copyText" />
      </DesignSystemProvider>,
    );
    expect(screen.getByText('Copy')).toBeInTheDocument();
    await userEvent.click(screen.getByText('Copy'));
    expect(screen.getByRole('tooltip', { name: 'Copied' })).toBeInTheDocument();
    expect(global.navigator.clipboard.writeText).toHaveBeenCalledWith('copyText');
  });
});
