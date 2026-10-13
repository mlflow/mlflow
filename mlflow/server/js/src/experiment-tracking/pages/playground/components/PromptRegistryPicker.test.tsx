import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import { render, screen, waitFor } from '@testing-library/react';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { RegisteredPromptsApi } from '../../prompts/api';
import { PromptRegistryPicker } from './PromptRegistryPicker';

// DialogCombobox masks elements behind pointer-events checks; disable so userEvent can click through.
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const mockListWith = (names: string[]) => {
  jest
    .spyOn(RegisteredPromptsApi, 'listRegisteredPrompts')
    .mockResolvedValue({ registered_models: names.map((name) => ({ name }) as any) });
};

const mockVersionWith = (
  promptName: string,
  type: 'chat' | 'text',
  text: string,
  options: { versionLabel?: string; modelConfig?: Record<string, unknown>; responseFormat?: string } = {},
) => {
  const { versionLabel = '1', modelConfig, responseFormat } = options;
  const tags: { key: string; value: string }[] = [
    { key: '_mlflow_prompt_type', value: type },
    { key: 'mlflow.prompt.text', value: text },
  ];
  if (modelConfig !== undefined) {
    tags.push({ key: '_mlflow_prompt_model_config', value: JSON.stringify(modelConfig) });
  }
  if (responseFormat !== undefined) {
    tags.push({ key: '_mlflow_prompt_response_format', value: responseFormat });
  }
  jest
    .spyOn(RegisteredPromptsApi, 'getPromptDetails')
    .mockResolvedValue({ registered_model: { name: promptName } as any });
  jest.spyOn(RegisteredPromptsApi, 'getPromptVersions').mockResolvedValue({
    model_versions: [{ name: promptName, version: versionLabel, tags } as any],
  });
};

const renderPicker = ({ visible = true }: { visible?: boolean } = {}) => {
  const onCancel = jest.fn();
  const onLoad = jest.fn();
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <PromptRegistryPicker visible={visible} onCancel={onCancel} onLoad={onLoad} />
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onCancel, onLoad };
};

const selectFromCombobox = async (label: RegExp, optionText: string) => {
  await userEvent.click(await screen.findByRole('combobox', { name: label }));
  await userEvent.click(await screen.findByText(optionText));
};

describe('PromptRegistryPicker', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the modal with both pickers and a disabled Load button', async () => {
    mockListWith([]);
    renderPicker();

    await waitFor(() => {
      expect(screen.getByText('Load prompt from registry')).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: /^load$/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /^cancel$/i })).toBeInTheDocument();
  });

  it('calls onCancel when the Cancel button is clicked', async () => {
    mockListWith([]);
    const { onCancel } = renderPicker();

    await waitFor(() => expect(screen.getByRole('button', { name: /^cancel$/i })).toBeInTheDocument());
    await userEvent.click(screen.getByRole('button', { name: /^cancel$/i }));
    expect(onCancel).toHaveBeenCalled();
  });

  it('loads a chat-typed prompt with the full payload on Load', async () => {
    mockListWith(['chat-prompt']);
    mockVersionWith(
      'chat-prompt',
      'chat',
      JSON.stringify([
        { role: 'system', content: 'You are concise.' },
        { role: 'user', content: 'Summarize: {{ text }}' },
      ]),
    );

    const { onLoad } = renderPicker();

    await selectFromCombobox(/prompt/i, 'chat-prompt');
    await selectFromCombobox(/version/i, 'v1');

    await userEvent.click(screen.getByRole('button', { name: /^load$/i }));

    await waitFor(() => {
      expect(onLoad).toHaveBeenCalledWith({
        messages: [
          { role: 'system', content: 'You are concise.' },
          { role: 'user', content: 'Summarize: {{ text }}' },
        ],
        settings: null,
        promptName: 'chat-prompt',
        versionLabel: '1',
        promptType: 'chat',
      });
    });
  });

  it('loads a text-typed prompt as a single user message and shows the info alert', async () => {
    mockListWith(['text-prompt']);
    mockVersionWith('text-prompt', 'text', 'Translate to French: {{ text }}');

    const { onLoad } = renderPicker();

    await selectFromCombobox(/prompt/i, 'text-prompt');
    await selectFromCombobox(/version/i, 'v1');

    await waitFor(() => {
      expect(screen.getByText(/text-typed prompt.*single user message/i)).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: /^load$/i }));

    await waitFor(() => {
      expect(onLoad).toHaveBeenCalledWith({
        messages: [{ role: 'user', content: 'Translate to French: {{ text }}' }],
        settings: null,
        promptName: 'text-prompt',
        versionLabel: '1',
        promptType: 'text',
      });
    });
  });

  it('renders the preview block once a version is selected (chat prompt, multiple roles)', async () => {
    mockListWith(['chat-prompt']);
    mockVersionWith(
      'chat-prompt',
      'chat',
      JSON.stringify([
        { role: 'system', content: 'You are concise.' },
        { role: 'user', content: 'Summarize: {{ text }}' },
      ]),
    );

    renderPicker();

    await selectFromCombobox(/prompt/i, 'chat-prompt');
    await selectFromCombobox(/version/i, 'v1');

    await waitFor(() => {
      expect(screen.getByText('Preview')).toBeInTheDocument();
    });
    expect(screen.getByText('system')).toBeInTheDocument();
    expect(screen.getByText('user')).toBeInTheDocument();
    expect(screen.getByText('You are concise.')).toBeInTheDocument();
    expect(screen.getByText('Summarize: {{ text }}')).toBeInTheDocument();
  });

  it('shows the empty-settings note in the preview when the version has no stored config', async () => {
    mockListWith(['chat-prompt']);
    mockVersionWith('chat-prompt', 'chat', JSON.stringify([{ role: 'user', content: 'Hi' }]));

    renderPicker();

    await selectFromCombobox(/prompt/i, 'chat-prompt');
    await selectFromCombobox(/version/i, 'v1');

    await waitFor(() => {
      expect(screen.getByText('No settings stored with this version')).toBeInTheDocument();
    });
  });

  it('shows the settings preview when the version has stored model config and a response format', async () => {
    mockListWith(['chat-prompt']);
    mockVersionWith('chat-prompt', 'chat', JSON.stringify([{ role: 'user', content: 'Hi' }]), {
      modelConfig: { temperature: 0.7, top_p: 0.9, stop_sequences: ['FIN'] },
      responseFormat: '{"type":"object"}',
    });

    renderPicker();

    await selectFromCombobox(/prompt/i, 'chat-prompt');
    await selectFromCombobox(/version/i, 'v1');

    await waitFor(() => {
      expect(screen.getByText('Temperature:')).toBeInTheDocument();
    });
    expect(screen.getByText('0.7')).toBeInTheDocument();
    expect(screen.getByText('Top P:')).toBeInTheDocument();
    expect(screen.getByText('0.9')).toBeInTheDocument();
    expect(screen.getByText('Stop Sequences:')).toBeInTheDocument();
    expect(screen.getByText('FIN')).toBeInTheDocument();
    expect(screen.getByText('Response format:')).toBeInTheDocument();
    expect(screen.getByText('JSON schema')).toBeInTheDocument();
  });

  it('emits a payload with mapped settings when the version has stored config', async () => {
    mockListWith(['chat-prompt']);
    mockVersionWith('chat-prompt', 'chat', JSON.stringify([{ role: 'user', content: 'Hi' }]), {
      modelConfig: { temperature: 0.7, stop_sequences: ['FIN'] },
      responseFormat: '{"type":"object"}',
    });

    const { onLoad } = renderPicker();

    await selectFromCombobox(/prompt/i, 'chat-prompt');
    await selectFromCombobox(/version/i, 'v1');

    await userEvent.click(screen.getByRole('button', { name: /^load$/i }));

    await waitFor(() => {
      expect(onLoad).toHaveBeenCalledWith({
        messages: [{ role: 'user', content: 'Hi' }],
        settings: {
          params: { temperature: 0.7, stop: ['FIN'] },
          responseFormatType: 'json_schema',
          responseFormatSchemaText: '{"type":"object"}',
        },
        promptName: 'chat-prompt',
        versionLabel: '1',
        promptType: 'chat',
      });
    });
  });
});
