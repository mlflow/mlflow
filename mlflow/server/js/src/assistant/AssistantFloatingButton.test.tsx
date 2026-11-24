import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { AssistantFloatingButton } from './AssistantFloatingButton';

const mockOpenPanel = jest.fn();
let mockAssistant: { isLocalServer: boolean; canUseAssistant: boolean; isPanelOpen: boolean; openPanel: jest.Mock };
let mockObstructionWidth: number;
let mockObstructionHeight: number;

jest.mock('./AssistantContext', () => ({
  useAssistant: () => mockAssistant,
}));

jest.mock('./useFloatingObstruction', () => ({
  useFloatingObstructionWidth: () => mockObstructionWidth,
  useFloatingObstructionHeight: () => mockObstructionHeight,
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
    mockAssistant = { isLocalServer: true, canUseAssistant: true, isPanelOpen: false, openPanel: mockOpenPanel };
    mockObstructionWidth = 0;
    mockObstructionHeight = 0;
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

  test('does not auto-open or render on a remote server without remote access', () => {
    mockAssistant = { isLocalServer: false, canUseAssistant: false, isPanelOpen: false, openPanel: mockOpenPanel };
    renderFab();
    expect(mockOpenPanel).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'MLflow Assistant' })).not.toBeInTheDocument();
  });

  test('does not render when canUseAssistant is false even if isLocalServer is true', () => {
    // Verifies the render guard uses canUseAssistant, not isLocalServer.
    mockAssistant = { isLocalServer: true, canUseAssistant: false, isPanelOpen: false, openPanel: mockOpenPanel };
    renderFab();
    expect(screen.queryByRole('button', { name: 'MLflow Assistant' })).not.toBeInTheDocument();
  });

  test('renders but does not auto-open on a remote server with remote access enabled', () => {
    // The button renders because canUseAssistant=true.
    // Auto-open stays local-only (gated on isLocalServer in the useEffect) — remote users should
    // not get a surprise panel pop-up on first load; they click the FAB themselves.
    mockAssistant = { isLocalServer: false, canUseAssistant: true, isPanelOpen: false, openPanel: mockOpenPanel };
    renderFab();
    expect(mockOpenPanel).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'MLflow Assistant' })).toBeInTheDocument();
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

  test('lifts (stays visible) when a bottom-pinned action bar is registered', () => {
    mockObstructionHeight = 80;
    renderFab();
    // The button rises above the bar rather than hiding; the bottom-inset math runs with a
    // real number (regression guard: an undefined height would make `bottom` NaN).
    expect(screen.getByRole('button', { name: 'MLflow Assistant' })).toBeInTheDocument();
  });
});
