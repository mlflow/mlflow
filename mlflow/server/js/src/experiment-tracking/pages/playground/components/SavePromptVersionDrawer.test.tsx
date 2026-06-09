import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { RegisteredPromptsApi } from '../../prompts/api';
import { SavePromptVersionDrawer } from './SavePromptVersionDrawer';
import type { ChatMessage } from '../types';

const CONTENT_TAG = 'mlflow.prompt.text';
const TYPE_TAG = '_mlflow_prompt_type';
const MODEL_CONFIG_TAG = '_mlflow_prompt_model_config';

const DEFAULT_MESSAGES: ChatMessage[] = [
  { role: 'system', content: 'You are concise.' },
  { role: 'user', content: 'Summarize: {{ text }}' },
];

const renderDrawer = (props: Partial<React.ComponentProps<typeof SavePromptVersionDrawer>> = {}) => {
  const onCancel = jest.fn();
  const onSaved = jest.fn();
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <SavePromptVersionDrawer
            visible
            onCancel={onCancel}
            messages={DEFAULT_MESSAGES}
            params={{}}
            responseFormatType="text"
            responseFormatSchemaText=""
            onSaved={onSaved}
            {...props}
          />
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onCancel, onSaved };
};

describe('SavePromptVersionDrawer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest
      .spyOn(RegisteredPromptsApi, 'createRegisteredPromptVersion')
      .mockResolvedValue({ model_version: { version: '2' } as any });
    jest.spyOn(RegisteredPromptsApi, 'createRegisteredPrompt').mockResolvedValue({});
  });

  it('disables Save and warns when there are no non-empty messages', async () => {
    renderDrawer({ messages: [{ role: 'user', content: '   ' }] });
    await waitFor(() => expect(screen.getByText('Save prompt to registry')).toBeInTheDocument());
    expect(screen.getByText(/add at least one non-empty message/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /save version/i })).toBeDisabled();
  });

  it('calls onCancel when Cancel is clicked', async () => {
    const { onCancel } = renderDrawer();
    await userEvent.click(screen.getByRole('button', { name: /^cancel$/i }));
    expect(onCancel).toHaveBeenCalled();
  });

  it('saves a new version of the loaded prompt with the serialized chat messages', async () => {
    const { onSaved } = renderDrawer({ loadedPromptName: 'greeter' });

    await userEvent.click(screen.getByRole('button', { name: /save version/i }));

    await waitFor(() => {
      expect(RegisteredPromptsApi.createRegisteredPromptVersion).toHaveBeenCalled();
    });
    const [name, tags] = jest.mocked(RegisteredPromptsApi.createRegisteredPromptVersion).mock.calls[0];
    expect(name).toBe('greeter');
    expect(tags).toEqual(
      expect.arrayContaining([
        { key: CONTENT_TAG, value: JSON.stringify(DEFAULT_MESSAGES) },
        { key: TYPE_TAG, value: 'chat' },
      ]),
    );
    // No new prompt entity is created when appending to an existing prompt.
    expect(RegisteredPromptsApi.createRegisteredPrompt).not.toHaveBeenCalled();
    await waitFor(() => expect(onSaved).toHaveBeenCalledWith({ name: 'greeter', version: '2' }));
  });

  it('requires a name and creates a brand-new prompt when none is loaded', async () => {
    const { onSaved } = renderDrawer();

    // Without a name the Save button stays disabled.
    expect(screen.getByRole('button', { name: /save version/i })).toBeDisabled();

    await userEvent.type(screen.getByPlaceholderText(/provide a unique prompt name/i), 'brand-new');
    await userEvent.click(screen.getByRole('button', { name: /save version/i }));

    await waitFor(() => {
      expect(RegisteredPromptsApi.createRegisteredPrompt).toHaveBeenCalledWith('brand-new', []);
    });
    expect(RegisteredPromptsApi.createRegisteredPromptVersion).toHaveBeenCalled();
    const [name] = jest.mocked(RegisteredPromptsApi.createRegisteredPromptVersion).mock.calls[0];
    expect(name).toBe('brand-new');
    await waitFor(() => expect(onSaved).toHaveBeenCalledWith({ name: 'brand-new', version: '2' }));
  });

  it('omits model settings when the include-settings checkbox is unchecked', async () => {
    renderDrawer({ loadedPromptName: 'greeter', params: { temperature: 0.5 } });

    // Settings are included by default, so the config tag is present.
    await userEvent.click(screen.getByRole('checkbox', { name: /save model settings/i }));
    await userEvent.click(screen.getByRole('button', { name: /save version/i }));

    await waitFor(() => expect(RegisteredPromptsApi.createRegisteredPromptVersion).toHaveBeenCalled());
    const [, tags] = jest.mocked(RegisteredPromptsApi.createRegisteredPromptVersion).mock.calls[0];
    expect((tags ?? []).some((tag) => tag.key === MODEL_CONFIG_TAG)).toBe(false);
  });

  it('includes model settings by default', async () => {
    renderDrawer({ loadedPromptName: 'greeter', params: { temperature: 0.5 } });

    await userEvent.click(screen.getByRole('button', { name: /save version/i }));

    await waitFor(() => expect(RegisteredPromptsApi.createRegisteredPromptVersion).toHaveBeenCalled());
    const [, tags] = jest.mocked(RegisteredPromptsApi.createRegisteredPromptVersion).mock.calls[0];
    expect(tags).toEqual(
      expect.arrayContaining([{ key: MODEL_CONFIG_TAG, value: JSON.stringify({ temperature: 0.5 }) }]),
    );
  });
});
