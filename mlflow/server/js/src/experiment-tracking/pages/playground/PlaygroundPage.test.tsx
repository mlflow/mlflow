import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

  it('forwards typed sampling parameters into the chat completion request', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    const endpointInput = await screen.findByTestId('endpoint-selector-test-input');
    await userEvent.type(endpointInput, 'my-endpoint');

    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello there');
    await userEvent.type(screen.getByLabelText('Temperature'), '1.5');

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

    // The Variables panel now renders an input for the detected variable.
    const variableInput = await screen.findByLabelText('text');
    await userEvent.type(variableInput, 'Hello world');

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
