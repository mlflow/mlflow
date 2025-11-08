import React from 'react';
import { CopyButton } from './CopyButton';
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
    renderWithIntl(<CopyButton copyText="copyText" />);
    expect(screen.getByText('Copy')).toBeInTheDocument();
    await userEvent.click(screen.getByText('Copy'));
    expect(screen.getByText('Copied')).toBeInTheDocument();
    expect(global.navigator.clipboard.writeText).toHaveBeenCalledWith('copyText');
  });
});
