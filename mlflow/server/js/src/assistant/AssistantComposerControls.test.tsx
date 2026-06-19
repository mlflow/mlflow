import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';

import { AssistantComposerControls } from './AssistantComposerControls';
import { updateConfig } from './AssistantService';
import { useAssistantConfigQuery } from './hooks/useAssistantConfigQuery';

jest.mock('./AssistantService', () => ({
  updateConfig: jest.fn(),
}));
jest.mock('./hooks/useAssistantConfigQuery', () => ({
  useAssistantConfigQuery: jest.fn(),
}));

const mockUpdateConfig = jest.mocked(updateConfig);
const mockUseConfig = jest.mocked(useAssistantConfigQuery);
const refetch = jest.fn<() => Promise<unknown>>();

type ConfigResult = ReturnType<typeof useAssistantConfigQuery>;

const gatewayConfig = {
  providers: {
    mlflow_gateway: {
      model: 'glm-4.7',
      selected: true,
      permissions: { allow_edit_files: true, allow_read_docs: true, full_access: false },
    },
  },
  projects: {},
};

const setConfig = (config: unknown) => mockUseConfig.mockReturnValue({ config, refetch } as unknown as ConfigResult);

const render = (onOpenSettings: () => void = jest.fn()) =>
  renderWithIntl(
    <DesignSystemProvider>
      <AssistantComposerControls onOpenSettings={onOpenSettings} />
    </DesignSystemProvider>,
  );

describe('AssistantComposerControls', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUpdateConfig.mockResolvedValue({} as Awaited<ReturnType<typeof updateConfig>>);
    refetch.mockResolvedValue({});
    setConfig(gatewayConfig);
  });

  test('shows the selected model name in the model pill', () => {
    render();
    expect(screen.getByText('glm-4.7')).toBeInTheDocument();
  });

  test('falls back to the provider label when the model is "default"', () => {
    setConfig({
      providers: {
        claude_code: {
          model: 'default',
          selected: true,
          permissions: { allow_edit_files: true, allow_read_docs: true, full_access: false },
        },
      },
      projects: {},
    });
    render();
    expect(screen.getByText('Claude Code')).toBeInTheDocument();
  });

  test('opens Model settings when the model pill is clicked', async () => {
    const user = userEvent.setup();
    const onOpenSettings = jest.fn();
    render(onOpenSettings);
    await user.click(screen.getByRole('button', { name: 'Change model in Settings' }));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  test('the model pill opens Model settings for an Ollama provider too', async () => {
    setConfig({
      providers: {
        ollama: {
          model: 'llama3.2',
          selected: true,
          base_url: 'http://localhost:11434',
          permissions: { allow_edit_files: true, allow_read_docs: true, full_access: false },
        },
      },
      projects: {},
    });
    const user = userEvent.setup();
    const onOpenSettings = jest.fn();
    render(onOpenSettings);
    await user.click(screen.getByRole('button', { name: 'Change model in Settings' }));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  test('renders nothing when no provider is selected', () => {
    setConfig({ providers: {}, projects: {} });
    const { container } = render();
    expect(container).toBeEmptyDOMElement();
  });

  test('changing execution mode persists merged permissions without a selected flag', async () => {
    const user = userEvent.setup();
    render();
    // full_access is false, so the pill reads "Read-only"; open it and switch to Full access.
    await user.click(screen.getByRole('button', { name: 'Read-only' }));
    await user.click(screen.getByRole('menuitemradio', { name: 'Full access' }));
    await waitFor(() =>
      expect(mockUpdateConfig).toHaveBeenCalledWith({
        providers: {
          mlflow_gateway: {
            permissions: { allow_edit_files: true, allow_read_docs: true, full_access: true },
          },
        },
      }),
    );
  });
});
