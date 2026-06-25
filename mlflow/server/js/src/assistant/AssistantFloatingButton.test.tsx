import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { AssistantFloatingButton } from './AssistantFloatingButton';

const mockOpenPanel = jest.fn();
let mockAssistant: { isLocalServer: boolean; isPanelOpen: boolean; openPanel: jest.Mock };
let mockObstructionWidth: number;

jest.mock('./AssistantContext', () => ({
  useAssistant: () => mockAssistant,
}));

jest.mock('./useFloatingObstruction', () => ({
  useFloatingObstructionWidth: () => mockObstructionWidth,
}));

jest.mock('../telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: jest.fn(() => jest.fn()),
}));

const renderFab = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <AssistantFloatingButton />
    </DesignSystemProvider>,
  );

describe('AssistantFloatingButton', () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockOpenPanel.mockClear();
    mockAssistant = { isLocalServer: true, isPanelOpen: false, openPanel: mockOpenPanel };
    mockObstructionWidth = 0;
  });

  test('auto-opens the panel once on first load', () => {
    renderFab();
    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'MLflow Assistant' })).toBeInTheDocument();
  });

  test('opens the panel when clicked', async () => {
    renderFab();
    // Ignore the first-load auto-open so we isolate the click.
    mockOpenPanel.mockClear();
    await userEvent.click(screen.getByRole('button', { name: 'MLflow Assistant' }));
    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
  });

  test('auto-opens only once across reloads', () => {
    const { unmount } = renderFab();
    expect(mockOpenPanel).toHaveBeenCalledTimes(1);

    unmount();
    mockOpenPanel.mockClear();
    // localStorage is intentionally not cleared here — the persisted flag should suppress re-open.
    renderFab();
    expect(mockOpenPanel).not.toHaveBeenCalled();
  });

  test('does not auto-open or render on a remote server', () => {
    mockAssistant.isLocalServer = false;
    renderFab();
    expect(mockOpenPanel).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'MLflow Assistant' })).not.toBeInTheDocument();
  });

  test('does not auto-open when the panel is already open', () => {
    mockAssistant.isPanelOpen = true;
    renderFab();
    expect(mockOpenPanel).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'MLflow Assistant' })).not.toBeInTheDocument();
  });

  test('stays visible (repositioned) when a right-side surface is open', () => {
    mockObstructionWidth = 600;
    renderFab();
    expect(screen.getByRole('button', { name: 'MLflow Assistant' })).toBeInTheDocument();
  });
});
