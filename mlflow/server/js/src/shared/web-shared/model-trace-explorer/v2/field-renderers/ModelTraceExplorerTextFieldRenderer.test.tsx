import { describe, expect, it, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { render } from '@databricks/testing-library';
import { ProvidersWrapper } from '../../../test-utils/testUtilProviderWrappers';
import { ModelTraceExplorerTextFieldRenderer } from './ModelTraceExplorerTextFieldRenderer';

const mockCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({
  useClipboard: () => ({ copy: mockCopy }),
}));

describe('ModelTraceExplorerTextFieldRenderer', () => {
  it('copies the full value when the displayed scalar is truncated', async () => {
    const value = 'a'.repeat(500);
    render(<ModelTraceExplorerTextFieldRenderer title="attribute" value={value} />, { wrapper: ProvidersWrapper });

    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));

    expect(mockCopy).toHaveBeenCalledWith(value);
  });
});
