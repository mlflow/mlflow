import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../common/utils/TestUtils.react18';
import SettingsPage from './SettingsPage';
import { DesignSystemProvider } from '@databricks/design-system';
import { DarkThemeProvider } from '../common/contexts/DarkThemeContext';
import { MemoryRouter, Route, Routes } from '../common/utils/RoutingUtils';

import { fetchEndpointRaw } from '../common/utils/FetchUtils';
import { useTelemetryConfig } from '../telemetry/hooks/useTelemetryConfig';
import { telemetryClient } from '../telemetry';
import { TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION } from '../telemetry/utils';
import { setLocalStorageItem } from '../shared/web-shared/hooks/useLocalStorage';

jest.mock('../telemetry/hooks/useTelemetryConfig');
jest.mock('../telemetry', () => ({
  telemetryClient: { start: jest.fn(), shutdown: jest.fn() },
}));

jest.mock('../common/utils/FetchUtils', () => ({
  fetchEndpointRaw: jest.fn(() => Promise.resolve()),
  HTTPMethods: { POST: 'POST', GET: 'GET' },
}));

jest.mock('./webhooksApi', () => ({
  WebhooksApi: {
    listWebhooks: jest.fn(() => Promise.resolve({ webhooks: [] })),
    createWebhook: jest.fn(() => Promise.resolve({})),
    updateWebhook: jest.fn(() => Promise.resolve({})),
    deleteWebhook: jest.fn(() => Promise.resolve()),
    testWebhook: jest.fn(() => Promise.resolve({ result: { success: true } })),
  },
}));

jest.mock('../gateway/pages/ApiKeysPage', () => ({
  __esModule: true,
  ApiKeysPageInner: () => <div data-testid="api-keys-settings-embed" />,
  default: () => null,
}));
const mockFetchEndpointRaw = jest.mocked(fetchEndpointRaw);

describe('SettingsPage', () => {
  const renderComponent = (initialEntry = '/settings/general') =>
    renderWithIntl(
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route
            path="/settings/:section"
            element={
              <DesignSystemProvider>
                <DarkThemeProvider setIsDarkTheme={() => {}}>
                  <SettingsPage />
                </DarkThemeProvider>
              </DesignSystemProvider>
            }
          />
        </Routes>
      </MemoryRouter>,
    );

  beforeEach(() => {
    jest.clearAllMocks();
    jest
      .mocked(useTelemetryConfig)
      .mockReturnValue({ data: { disable_ui_telemetry: false } } as ReturnType<typeof useTelemetryConfig>);
    setLocalStorageItem(TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION, false, true);
  });

  it('forces telemetry off when the server disables it', async () => {
    jest.mocked(useTelemetryConfig).mockReturnValue({
      data: { disable_ui_telemetry: true },
    } as ReturnType<typeof useTelemetryConfig>);
    renderComponent();

    const toggle = screen.getByRole('switch', { name: 'Off' });
    expect(toggle).toBeDisabled();
    expect(toggle).not.toBeChecked();
    await userEvent.click(toggle);
    expect(telemetryClient.start).not.toHaveBeenCalled();
  });

  it('still allows browser opt-out when server configuration is unavailable', async () => {
    jest.mocked(useTelemetryConfig).mockReturnValue({ data: undefined } as ReturnType<typeof useTelemetryConfig>);
    renderComponent();

    const toggle = screen.getByRole('switch', { name: 'On' });
    expect(toggle).toBeEnabled();
    await userEvent.click(toggle);
    expect(toggle).not.toBeChecked();
    expect(telemetryClient.shutdown).toHaveBeenCalled();
  });

  it('allows toggling telemetry when the server enables it', async () => {
    renderComponent();

    const toggle = screen.getByRole('switch', { name: 'On' });
    expect(toggle).toBeEnabled();
    expect(toggle).toBeChecked();
    await userEvent.click(toggle);
    expect(toggle).not.toBeChecked();
    expect(telemetryClient.shutdown).toHaveBeenCalled();
    await userEvent.click(toggle);
    expect(toggle).toBeChecked();
    expect(telemetryClient.start).toHaveBeenCalled();
  });

  it('preserves a browser opt-out when the server enables telemetry', () => {
    setLocalStorageItem(TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION, false, false);
    renderComponent();

    const toggle = screen.getByRole('switch', { name: 'Off' });
    expect(toggle).toBeEnabled();
    expect(toggle).not.toBeChecked();
  });

  it('shows demo data controls under General section', async () => {
    renderComponent('/settings/general');

    expect(await screen.findByText('Clear all demo data')).toBeInTheDocument();
  });

  it('opens LLM Connections from the URL path and embeds API keys', async () => {
    renderComponent('/settings/llm-connections');

    expect(await screen.findByTestId('api-keys-settings-embed')).toBeInTheDocument();
  });

  it('calls fetchEndpointRaw with the demo delete endpoint when clearing demo data', async () => {
    renderComponent('/settings/general');

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
