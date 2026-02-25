import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../common/utils/TestUtils.react18';
import SettingsPage from './SettingsPage';
import { DesignSystemProvider } from '@databricks/design-system';
import { DarkThemeProvider } from '../common/contexts/DarkThemeContext';

jest.mock('../common/utils/FetchUtils', () => ({
  fetchEndpointRaw: jest.fn(() => Promise.resolve()),
  HTTPMethods: { POST: 'POST', GET: 'GET' },
}));

import { fetchEndpointRaw } from '../common/utils/FetchUtils';
const mockFetchEndpointRaw = fetchEndpointRaw as jest.MockedFunction<typeof fetchEndpointRaw>;

describe('SettingsPage', () => {
  const renderComponent = () =>
    renderWithIntl(
      <DesignSystemProvider>
        <DarkThemeProvider setIsDarkTheme={() => {}}>
          <SettingsPage />
        </DarkThemeProvider>
      </DesignSystemProvider>,
    );

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('calls fetchEndpointRaw with the demo delete endpoint when clearing demo data', async () => {
    renderComponent();

    // Open the confirmation modal
    await userEvent.click(screen.getByText('Clear all demo data'));

    // Confirm deletion
    await userEvent.click(screen.getByText('Clear'));

    await waitFor(() => {
      expect(mockFetchEndpointRaw).toHaveBeenCalledWith({
        relativeUrl: 'ajax-api/3.0/mlflow/demo/delete',
        method: 'POST',
      });
    });
  });
});
