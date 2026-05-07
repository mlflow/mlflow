import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { render, renderHook, screen, waitFor } from '@testing-library/react';
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
  });

  it('renders the page header and the empty completion output by default', async () => {
    renderPlayground();

    await waitFor(() => {
      expect(screen.getByText('Playground')).toBeInTheDocument();
    });

    expect(screen.getByText('Submit a message to see the response here.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
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
