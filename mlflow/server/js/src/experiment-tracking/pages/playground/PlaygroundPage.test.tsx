import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { render, renderHook, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { GatewayApi } from '../../../gateway/api';
import { PlaygroundApi } from './api';
import PlaygroundPage from './PlaygroundPage';
import { useChatCompletionMutation } from './hooks/useChatCompletionMutation';

jest.mock('./components/EndpointPicker', () => ({
  EndpointPicker: ({ value, onChange }: { value?: string; onChange: (v: string) => void }) => (
    <div>
      <label htmlFor="endpoint-picker-test-input">Endpoint</label>
      <input
        id="endpoint-picker-test-input"
        data-testid="endpoint-picker-test-input"
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
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

const pickEndpoint = async (name: string) => {
  const input = await screen.findByTestId('endpoint-picker-test-input');
  await userEvent.clear(input);
  await userEvent.type(input, name);
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
  });

  it('renders the page header and a disabled Submit button by default', async () => {
    renderPlayground();

    await waitFor(() => {
      expect(screen.getByText('Playground')).toBeInTheDocument();
    });

    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /clear conversation/i })).toBeInTheDocument();
  });

  it('enables submit only when the message and endpoint are both present', async () => {
    renderPlayground();

    await waitFor(() => {
      expect(screen.getByText('Playground')).toBeInTheDocument();
    });

    const submit = screen.getByRole('button', { name: /submit/i });
    expect(submit).toBeDisabled();

    const textarea = screen.getByPlaceholderText('Type a message');
    await userEvent.type(textarea, 'Hello there');
    expect(submit).toBeDisabled();

    await pickEndpoint('my-endpoint');
    expect(submit).toBeEnabled();
  });

  it('appends the assistant response and a fresh user card after a successful submit', async () => {
    jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'Hello back!' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    await pickEndpoint('my-endpoint');
    const textarea = screen.getByPlaceholderText('Type a message');
    await userEvent.type(textarea, 'Hello there');

    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    await waitFor(() => {
      const assistantCard = screen.getByTestId('mlflow.playground.prompt_input.message.assistant');
      expect(within(assistantCard).getByText('Hello back!')).toBeInTheDocument();
    });

    const userCards = screen.getAllByTestId('mlflow.playground.prompt_input.message.user');
    expect(userCards).toHaveLength(2);
    expect(within(userCards[1]).getByPlaceholderText('Type a message')).toHaveValue('');
  });

  it('clears the conversation when the Clear button is pressed', async () => {
    renderPlayground();

    const textarea = screen.getByPlaceholderText('Type a message');
    await userEvent.type(textarea, 'Some draft');

    await userEvent.click(screen.getByRole('button', { name: /clear conversation/i }));

    const userCards = screen.getAllByTestId('mlflow.playground.prompt_input.message.user');
    expect(userCards).toHaveLength(1);
    expect(within(userCards[0]).getByPlaceholderText('Type a message')).toHaveValue('');
  });

  it('forwards typed parameters to the chat completion request', async () => {
    const chatCompletionSpy = jest.spyOn(PlaygroundApi, 'chatCompletion').mockResolvedValue({
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    });

    renderPlayground();

    await pickEndpoint('my-endpoint');
    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Hello there');

    const temperatureInput = screen.getByLabelText('Temperature');
    await userEvent.type(temperatureInput, '1.5');

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
