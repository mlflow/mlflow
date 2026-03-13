import { describe, test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import React from 'react';
import { fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { getDefaultHeaders } from '../../../common/utils/FetchUtils';
import { EndpointUsageModal } from './EndpointUsageModal';

const renderModal = (props: React.ComponentProps<typeof EndpointUsageModal>) =>
  renderWithDesignSystem(
    <QueryClientProvider client={new QueryClient()}>
      <EndpointUsageModal {...props} />
    </QueryClientProvider>,
  );

jest.mock('../../../common/utils/FetchUtils', () => {
  const actual = jest.requireActual<typeof import('../../../common/utils/FetchUtils')>(
    '../../../common/utils/FetchUtils',
  );
  const { matchPredefinedErrorFromResponse } = jest.requireActual<
    typeof import('../../../shared/web-shared/errors/PredefinedErrors')
  >('../../../shared/web-shared/errors/PredefinedErrors');
  const getDefaultHeadersMock = jest.fn((cookieStr: string) => actual.getDefaultHeaders(cookieStr));
  return {
    ...actual,
    getDefaultHeaders: getDefaultHeadersMock,
    fetchOrFail: jest.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      const doc = typeof global !== 'undefined' ? (global as { document?: { cookie?: string } }).document : undefined;
      const cookieString = doc && typeof doc.cookie === 'string' ? doc.cookie : '';
      const fetchOptions: RequestInit = {
        ...options,
        headers: {
          ...getDefaultHeadersMock(cookieString),
          ...options?.headers,
        },
      };
      const response = await fetch(input, fetchOptions);
      if (!response.ok) {
        throw matchPredefinedErrorFromResponse(response);
      }
      return response;
    }),
  };
});

describe('Try it tab', () => {
  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    endpointName: 'test-endpoint',
    baseUrl: 'http://localhost:5000',
  };

  let fetchSpy: ReturnType<typeof jest.spyOn>;

  const getRequestBodyTextarea = (): HTMLTextAreaElement => {
    const textareas = screen.getAllByRole('textbox');
    return textareas[0] as HTMLTextAreaElement;
  };

  beforeEach(() => {
    fetchSpy = jest.spyOn(global, 'fetch');
  });

  afterEach(() => {
    fetchSpy?.mockRestore();
  });

  test('sends request and displays response on success', async () => {
    const mockResponse = { output: 'Hello from the model' };
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
    });

    renderModal(defaultProps);
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy).toHaveBeenCalledWith(
      `${defaultProps.baseUrl}/gateway/${defaultProps.endpointName}/mlflow/invocations`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      }),
    );
    expect(await screen.findByText(/Hello from the model/)).toBeInTheDocument();
  });

  test('sends request with auth headers when auth is enabled', async () => {
    const authHeaders = {
      Authorization: 'Bearer test-token',
    } as ReturnType<typeof getDefaultHeaders>;
    jest.mocked(getDefaultHeaders).mockReturnValueOnce(authHeaders);
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ result: 'ok' })),
    });

    renderModal(defaultProps);
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [, options] = fetchSpy.mock.calls[0];
    expect(options?.headers).toMatchObject({
      'Content-Type': 'application/json',
      ...authHeaders,
    });
  });

  test('shows error when request body is invalid JSON', async () => {
    renderModal(defaultProps);
    const requestBody = getRequestBodyTextarea();
    await userEvent.clear(requestBody);
    await userEvent.type(requestBody, 'not valid json');
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(screen.getByText('Invalid JSON in request body')).toBeInTheDocument();
  });

  test('shows error and response body when request fails with HTTP error', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      text: () => Promise.resolve('Server error details'),
    });

    renderModal(defaultProps);
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Internal server error')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Server error details')).toBeInTheDocument();
  });

  test('shows error when network request fails', async () => {
    fetchSpy.mockRejectedValueOnce(new Error('Network failure'));

    renderModal(defaultProps);
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Network failure')).toBeInTheDocument();
  });

  test('switching to Passthrough APIs tab shows provider selector and updates request body', async () => {
    renderModal(defaultProps);
    expect(getRequestBodyTextarea().value).toContain('messages');

    await userEvent.click(screen.getByText('Passthrough APIs'));
    expect(screen.getByText('Provider')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /OpenAI/ })).toBeInTheDocument();
    expect(getRequestBodyTextarea().value).toContain('model');
  });

  test('switching from Passthrough APIs back to Unified APIs resets request body to unified format', async () => {
    renderModal(defaultProps);
    await userEvent.click(screen.getByText('Passthrough APIs'));
    expect(getRequestBodyTextarea().value).toContain('model');

    await userEvent.click(screen.getByText('Unified APIs'));
    expect(getRequestBodyTextarea().value).toContain('messages');
  });

  test('Unified OpenAI Chat Completions variant updates request body and sends to chat completions URL', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ choices: [{ message: { content: 'Hi' } }] })),
    });

    renderModal(defaultProps);
    await userEvent.click(screen.getByRole('radio', { name: 'OpenAI Chat Completions' }));

    expect(getRequestBodyTextarea().value).toContain('model');
    expect(getRequestBodyTextarea().value).toContain('messages');
    expect(getRequestBodyTextarea().value).toContain(defaultProps.endpointName);

    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy).toHaveBeenCalledWith(
      `${defaultProps.baseUrl}/gateway/mlflow/v1/chat/completions`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      }),
    );
    expect(await screen.findByText(/Hi/)).toBeInTheDocument();
  });

  test('provider selection in Try it passthrough updates request body', async () => {
    renderModal(defaultProps);
    await userEvent.click(screen.getByText('Passthrough APIs'));

    await userEvent.click(screen.getByRole('radio', { name: 'Anthropic' }));
    expect(getRequestBodyTextarea().value).toContain('max_tokens');
    expect(getRequestBodyTextarea().value).toContain('messages');

    await userEvent.click(screen.getByRole('radio', { name: 'Google Gemini' }));
    expect(getRequestBodyTextarea().value).toContain('contents');
  });

  test('state reset when switching API type clears response and error', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ result: 'old' })),
    });

    renderModal(defaultProps);
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
    expect(screen.getByText(/"result": "old"/)).toBeInTheDocument();

    await userEvent.click(screen.getByText('Passthrough APIs'));
    expect(screen.queryByText(/"result": "old"/)).not.toBeInTheDocument();
    expect(screen.queryByText('Invalid JSON')).not.toBeInTheDocument();
  });

  test('state reset when switching provider clears response and error', async () => {
    fetchSpy.mockRejectedValueOnce(new Error('First error'));

    renderModal(defaultProps);
    await userEvent.click(screen.getByText('Passthrough APIs'));
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
    expect(screen.getByText('First error')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'Anthropic' }));
    expect(screen.queryByText('First error')).not.toBeInTheDocument();
  });

  test('state reset when modal opens clears previous response and error', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ data: 'previous' })),
    });

    const queryClient = new QueryClient();
    const { rerender } = renderWithDesignSystem(
      <QueryClientProvider client={queryClient}>
        <EndpointUsageModal {...defaultProps} open />
      </QueryClientProvider>,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
    expect(screen.getByText(/"data": "previous"/)).toBeInTheDocument();

    rerender(
      <QueryClientProvider client={queryClient}>
        <EndpointUsageModal {...defaultProps} open={false} />
      </QueryClientProvider>,
    );
    rerender(
      <QueryClientProvider client={queryClient}>
        <EndpointUsageModal {...defaultProps} open />
      </QueryClientProvider>,
    );

    expect(screen.queryByText(/"data": "previous"/)).not.toBeInTheDocument();
    expect(screen.getByText('Send request')).toBeInTheDocument();
  });

  test('Reset example restores default request body and clears response and error', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ out: 1 })),
    });

    renderModal(defaultProps);
    const requestBody = getRequestBodyTextarea();
    await userEvent.clear(requestBody);
    fireEvent.change(requestBody, { target: { value: '{"custom": true}' } });
    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
    expect(await screen.findByText(/"out": 1/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Reset example' }));
    expect(getRequestBodyTextarea().value).toContain('messages');
    expect(screen.queryByText(/"out": 1/)).not.toBeInTheDocument();
  });
});
