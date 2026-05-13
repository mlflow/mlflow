import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event';
import React from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { GatewayApi } from '../../../gateway/api';
import { RegisteredPromptsApi } from '../prompts/api';
import { PlaygroundApi } from './api';
import PlaygroundPage from './PlaygroundPage';
import { useChatCompletionMutation } from './hooks/useChatCompletionMutation';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

// Drawer + DialogCombobox use Radix overlays that cover their triggers; disable userEvent's
// pointer-events check so clicks register through them.
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const openSettingsDrawer = () => userEvent.click(screen.getByRole('button', { name: /open model parameters/i }));
const openVariablesDrawer = () => userEvent.click(screen.getByRole('button', { name: /open variable values/i }));

// Drawers are focus-trapping Radix dialogs — Submit isn't reachable until the drawer is closed.
const closeDrawer = () => userEvent.keyboard('{Escape}');

jest.mock('../../components/EndpointSelector', () => ({
  EndpointSelector: ({
    currentEndpointName,
    onEndpointSelect,
  }: {
    currentEndpointName?: string;
    onEndpointSelect: (name: string) => void;
  }) => (
    <input
      data-testid="endpoint-selector-test-input"
      value={currentEndpointName ?? ''}
      onChange={(event) => onEndpointSelect(event.target.value)}
    />
  ),
}));

const renderPlayground = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<PlaygroundPage />, {
    wrapper: ({ children }) => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={queryClient}>
            <TestRouter routes={[testRoute(<>{children}</>, '/'), testRoute(<div />, '*')]} initialEntries={['/']} />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>
    ),
  });
};

describe('PlaygroundPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(GatewayApi, 'listEndpoints').mockResolvedValue({
      endpoints: [
        {
          endpoint_id: 'ep-1',
          name: 'my-endpoint',
          model_mappings: [],
          created_at: 1700000000000,
          last_updated_at: 1700000000000,
        },
      ],
    });
    // PromptRegistryPicker mounts inside the page (Modal hidden by default).
    // Stub the registered-prompts list so the underlying hook doesn't hit the network.
    jest.spyOn(RegisteredPromptsApi, 'listRegisteredPrompts').mockResolvedValue({
      registered_models: [],
    });
  });

  it('renders the page header and the empty completion output by default', async () => {
    renderPlayground();

    await waitFor(() => {
      expect(screen.getByText('Playground')).toBeInTheDocument();
    });

    expect(screen.getByText('Submit a message to see the response here.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /load prompt from registry/i })).toBeInTheDocument();
  });

  it('keeps Submit disabled until both an endpoint and a non-empty message are present', async () => {
    renderPlayground();

    await waitFor(() => {
      expect(screen.getByText('Playground')).toBeInTheDocument();
    });

    const submit = screen.getByRole('button', { name: /submit/i });
    expect(submit).toBeDisabled();

    const textarea = screen.getByPlaceholderText('Type a message');
    await userEvent.type(textarea, 'Hello there');

    // Endpoint is still empty, so submit stays disabled.
    expect(submit).toBeDisabled();
  });

  it('keeps Submit disabled while any message textbox is empty', async () => {
    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello');

    // The Submit button is re-queried each time because the HoverCard wrapper
    // unmounts when there are no blockers, swapping the DOM node identity.
    expect(screen.getByRole('button', { name: /submit/i })).toBeEnabled();

    // Add a second message and leave it empty — Submit should disable again.
    await userEvent.click(screen.getByRole('button', { name: /add message/i }));
    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
  });

  it('keeps Submit disabled while a template variable is unfilled, then enables once provided', async () => {
    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    // fireEvent.change avoids userEvent's `{{`-escaping rules for braces.
    fireEvent.change(screen.getByPlaceholderText('Type a message'), { target: { value: 'Hi {{ name }}' } });

    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();

    await openVariablesDrawer();
    await userEvent.type(await screen.findByLabelText('name'), 'Alice');
    await closeDrawer();

    expect(screen.getByRole('button', { name: /submit/i })).toBeEnabled();
  });

  it('keeps Submit disabled when tool choice is required but no tool definition is provided', async () => {
    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello');

    expect(screen.getByRole('button', { name: /submit/i })).toBeEnabled();

    await openSettingsDrawer();
    await userEvent.click(screen.getByRole('radio', { name: 'Required' }));
    await closeDrawer();

    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
  });

  it('keeps Submit disabled when tool choice is required and tools parses to an empty array', async () => {
    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello');

    await openSettingsDrawer();
    await userEvent.click(screen.getByRole('radio', { name: 'Required' }));
    fireEvent.change(screen.getByLabelText('JSON Tool Definition'), { target: { value: '[]' } });
    await closeDrawer();

    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
  });

  it('keeps Submit disabled when response_format is json_schema and the value is not a JSON object', async () => {
    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello');

    await openSettingsDrawer();
    await userEvent.click(screen.getByRole('radio', { name: 'JSON schema' }));
    fireEvent.change(screen.getByLabelText('Schema'), { target: { value: '42' } });
    await closeDrawer();

    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
  });

  it('forwards typed sampling parameters into the chat completion request', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');

    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello there');

    await openSettingsDrawer();
    await userEvent.type(screen.getByLabelText('Temperature'), '1.5');

    await closeDrawer();
    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(chatCompletionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'my-endpoint',
          temperature: 1.5,
          messages: [{ role: 'user', content: 'Hello there' }],
        }),
      );
    });
  });

  it('forwards advanced sampling params, stop sequences, tool_choice and response_format', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello there');

    await openSettingsDrawer();

    // Reveal and fill the advanced sampling block.
    await userEvent.click(screen.getByRole('button', { name: /advanced/i }));
    await userEvent.type(screen.getByLabelText('Top K'), '40');
    await userEvent.type(screen.getByLabelText('Presence penalty'), '0.5');
    await userEvent.type(screen.getByLabelText('Frequency penalty'), '0.5');
    fireEvent.change(screen.getByLabelText('Stop sequences'), { target: { value: 'STOP\nFIN' } });

    // Switch tool_choice to required and supply a valid tools array.
    await userEvent.click(screen.getByRole('radio', { name: 'Required' }));
    fireEvent.change(screen.getByLabelText('JSON Tool Definition'), {
      target: { value: '[{"type":"function","function":{"name":"echo"}}]' },
    });

    // Switch response format to json_object.
    await userEvent.click(screen.getByRole('radio', { name: 'JSON' }));

    await closeDrawer();
    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(chatCompletionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'my-endpoint',
          messages: [{ role: 'user', content: 'Hello there' }],
          top_k: 40,
          presence_penalty: 0.5,
          frequency_penalty: 0.5,
          stop: ['STOP', 'FIN'],
          tools: [{ type: 'function', function: { name: 'echo' } }],
          tool_choice: 'required',
          response_format: { type: 'json_object' },
        }),
      );
    });
  });

  it('wraps json_schema response_format in the OpenAI envelope on submit', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello there');

    await openSettingsDrawer();
    await userEvent.click(screen.getByRole('radio', { name: 'JSON schema' }));
    fireEvent.change(screen.getByLabelText('Schema'), {
      target: { value: '{"type":"object","properties":{"x":{"type":"string"}}}' },
    });
    await closeDrawer();
    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(chatCompletionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          response_format: {
            type: 'json_schema',
            json_schema: {
              name: 'response_schema',
              schema: { type: 'object', properties: { x: { type: 'string' } } },
              strict: true,
            },
          },
        }),
      );
    });
  });

  it('omits tools and tool_choice when toolChoice stays at none, even with tool text typed', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello there');

    // Pick Required just to reveal the textarea, type something, then revert to None.
    await openSettingsDrawer();
    await userEvent.click(screen.getByRole('radio', { name: 'Required' }));
    fireEvent.change(screen.getByLabelText('JSON Tool Definition'), {
      target: { value: '[{"type":"function","function":{"name":"echo"}}]' },
    });
    await userEvent.click(screen.getByRole('radio', { name: 'None' }));

    await closeDrawer();
    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(chatCompletionSpy).toHaveBeenCalled();
    });
    expect(chatCompletionSpy).toHaveBeenCalledWith(expect.not.objectContaining({ tools: expect.anything() }));
    expect(chatCompletionSpy).toHaveBeenCalledWith(expect.not.objectContaining({ tool_choice: expect.anything() }));
  });

  it('substitutes typed variable values into the request body and leaves the template intact', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');

    // Set the user message to a template — fireEvent.change avoids userEvent's
    // `{{`-escaping rules for braces.
    const template = 'Summarize: {{ text }}';
    fireEvent.change(screen.getByPlaceholderText('Type a message'), { target: { value: template } });

    // The Variables drawer now renders an input for the detected variable.
    await openVariablesDrawer();
    const variableInput = await screen.findByLabelText('text');
    await userEvent.type(variableInput, 'Hello world');

    await closeDrawer();
    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(chatCompletionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'my-endpoint',
          messages: [{ role: 'user', content: 'Summarize: Hello world' }],
        }),
      );
    });

    // The template in the message panel is preserved (un-substituted).
    expect(screen.getByPlaceholderText('Type a message')).toHaveValue(template);
  });

  const stubPromptVersion = (
    promptName: string,
    type: 'chat' | 'text',
    text: string,
    extras: { modelConfig?: Record<string, unknown>; responseFormat?: string } = {},
  ) => {
    jest.spyOn(RegisteredPromptsApi, 'listRegisteredPrompts').mockResolvedValue({
      registered_models: [{ name: promptName } as any],
    });
    const tags: { key: string; value: string }[] = [
      { key: '_mlflow_prompt_type', value: type },
      { key: 'mlflow.prompt.text', value: text },
    ];
    if (extras.modelConfig !== undefined) {
      tags.push({ key: '_mlflow_prompt_model_config', value: JSON.stringify(extras.modelConfig) });
    }
    if (extras.responseFormat !== undefined) {
      tags.push({ key: '_mlflow_prompt_response_format', value: extras.responseFormat });
    }
    jest
      .spyOn(RegisteredPromptsApi, 'getPromptDetails')
      .mockResolvedValue({ registered_model: { name: promptName } as any });
    jest.spyOn(RegisteredPromptsApi, 'getPromptVersions').mockResolvedValue({
      model_versions: [{ name: promptName, version: '1', tags } as any],
    });
  };

  const openRegistryAndLoad = async (promptName: string) => {
    await userEvent.click(screen.getByRole('button', { name: /load prompt from registry/i }));
    await userEvent.click(await screen.findByRole('combobox', { name: /prompt/i }));
    await userEvent.click(await screen.findByText(promptName));
    await userEvent.click(await screen.findByRole('combobox', { name: /version/i }));
    await userEvent.click(await screen.findByText('v1'));
    await userEvent.click(screen.getByRole('button', { name: /^load$/i }));
  };

  it('resets and applies settings as a unit when the loaded version carries stored config', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });
    stubPromptVersion('cfg-prompt', 'chat', JSON.stringify([{ role: 'user', content: 'Hi' }]), {
      modelConfig: { top_p: 0.5 },
    });

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');

    // Pre-fill a temperature that must be wiped out by the load (replace-as-unit).
    await openSettingsDrawer();
    await userEvent.type(screen.getByLabelText('Temperature'), '0.1');
    await closeDrawer();

    await openRegistryAndLoad('cfg-prompt');

    await waitFor(() => {
      expect(screen.getByText(/Loaded cfg-prompt v1 with settings/i)).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(chatCompletionSpy).toHaveBeenCalled();
    });
    expect(chatCompletionSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'my-endpoint',
        top_p: 0.5,
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    );
    expect(chatCompletionSpy).toHaveBeenCalledWith(expect.not.objectContaining({ temperature: expect.anything() }));
  });

  it('leaves existing playground settings untouched when the loaded version has no stored settings', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });
    stubPromptVersion('plain-prompt', 'text', 'Plain text');

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');

    await openSettingsDrawer();
    await userEvent.type(screen.getByLabelText('Temperature'), '0.1');
    await closeDrawer();

    await openRegistryAndLoad('plain-prompt');

    await waitFor(() => {
      expect(screen.getByText(/Loaded plain-prompt v1$/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(/with settings/i)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(chatCompletionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'my-endpoint',
          temperature: 0.1,
        }),
      );
    });
  });

  it('never auto-sets the endpoint from the loaded version provider/model_name', async () => {
    stubPromptVersion('cfg-prompt', 'chat', JSON.stringify([{ role: 'user', content: 'Hi' }]), {
      modelConfig: { provider: 'openai', model_name: 'gpt-4' },
    });

    renderPlayground();

    await openRegistryAndLoad('cfg-prompt');

    await waitFor(() => {
      expect(screen.getByText(/Loaded cfg-prompt v1 with settings/i)).toBeInTheDocument();
    });
    expect(screen.getByTestId('endpoint-selector-test-input')).toHaveValue('');
    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
  });

  it('surfaces the upstream message and HTTP status when chat completion rejects', async () => {
    jest
      .spyOn(PlaygroundApi, 'chatCompletion')
      .mockRejectedValue(Object.assign(new Error('Gemini rejected schema: foo'), { status: 400 }));

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello');

    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      expect(screen.getByText('Chat completion failed')).toBeInTheDocument();
    });
    expect(screen.getByText('HTTP 400 — Gemini rejected schema: foo')).toBeInTheDocument();
  });

  it('exposes a hook that posts to the chat completions API', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'Hello back!' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    });

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(() => useChatCompletionMutation(), { wrapper });

    result.current.mutate({
      model: 'my-endpoint',
      messages: [{ role: 'user', content: 'Hello there' }],
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(chatCompletionSpy).toHaveBeenCalledWith({
      model: 'my-endpoint',
      messages: [{ role: 'user', content: 'Hello there' }],
    });
    expect(result.current.data?.choices?.[0]?.message?.content).toBe('Hello back!');
  });
});
