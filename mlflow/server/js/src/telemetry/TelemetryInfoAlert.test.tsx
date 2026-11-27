import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../common/utils/TestUtils.react18';
import { MemoryRouter } from '../common/utils/RoutingUtils';
import { setLocalStorageItem } from '../shared/web-shared/hooks/useLocalStorage';
import { TelemetryInfoAlert } from './TelemetryInfoAlert';
import { useTelemetryConfig } from './hooks/useTelemetryConfig';
import { TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_KEY, TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_VERSION } from './utils';

jest.mock('./hooks/useTelemetryConfig');

describe('TelemetryInfoAlert', () => {
  const renderComponent = () =>
    renderWithDesignSystem(
      <MemoryRouter>
        <TelemetryInfoAlert />
      </MemoryRouter>,
    );

  beforeEach(() => {
    setLocalStorageItem(
      TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_KEY,
      TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_VERSION,
      false,
      false,
    );
  });

  it.each([true, false, undefined])('respects server disable setting %s', (disabled) => {
    jest.mocked(useTelemetryConfig).mockReturnValue({
      data: disabled === undefined ? undefined : { disable_ui_telemetry: disabled },
    } as ReturnType<typeof useTelemetryConfig>);
    renderComponent();

    if (disabled === false) {
      expect(screen.getByText('Information about UI telemetry')).toBeInTheDocument();
    } else {
      expect(screen.queryByText('Information about UI telemetry')).not.toBeInTheDocument();
    }
  });

  it('still allows dismissing the banner when telemetry is enabled', async () => {
    jest.mocked(useTelemetryConfig).mockReturnValue({
      data: { disable_ui_telemetry: false },
    } as ReturnType<typeof useTelemetryConfig>);
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: 'Close alert' }));
    expect(screen.queryByText('Information about UI telemetry')).not.toBeInTheDocument();
  });
});
