import { describe, test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { fireEvent } from '@testing-library/react';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { EndpointUsageModal } from './EndpointUsageModal';

describe('EndpointUsageModal', () => {
  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    endpointName: 'test-endpoint',
    baseUrl: 'http://localhost:5000',
  };

  let fetchSpy: jest.SpyInstance;

  beforeEach(() => {
    fetchSpy = jest.spyOn(global, 'fetch');
  });

  afterEach(() => {
    fetchSpy?.mockRestore();
  });

  test('renders modal with title when open', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    expect(screen.getByText('Query endpoint')).toBeInTheDocument();
  });

  test('does not render modal when closed', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} open={false} />);

    expect(screen.queryByText('Query endpoint')).not.toBeInTheDocument();
  });

  test('renders Try it tab by default', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    expect(screen.getByText('Send request')).toBeInTheDocument();
    expect(screen.getByText('Request')).toBeInTheDocument();
    expect(screen.getByText('Response')).toBeInTheDocument();
  });

  test('renders language selector in unified APIs tab', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByText('Unified APIs'));

    // Verify language selector is rendered with both options
    expect(screen.getByRole('radio', { name: /cURL/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Python/ })).toBeInTheDocument();

    // cURL is selected by default - shows curl content for both API sections
    expect(screen.getByRole('radio', { name: /cURL/ })).toBeChecked();
    const curlElements = screen.getAllByText(/curl -X POST/);
    expect(curlElements.length).toBe(2); // MLflow Invocations and OpenAI-Compatible both show cURL
  });

  test('shows code examples with endpoint name in unified APIs', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByText('Unified APIs'));

    // The endpoint name appears in multiple code examples, so use getAllByText
    const matchingElements = screen.getAllByText(/gateway\/test-endpoint\/mlflow\/invocations/);
    expect(matchingElements.length).toBeGreaterThan(0);
  });

  test('switches to passthrough APIs tab when clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByText('Passthrough APIs'));

    expect(screen.getByText('Provider')).toBeInTheDocument();
    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    expect(screen.getByText('Anthropic')).toBeInTheDocument();
    expect(screen.getByText('Google Gemini')).toBeInTheDocument();
  });

  test('shows OpenAI passthrough example by default in passthrough tab', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Passthrough APIs'));

    expect(
      screen.getByText(
        "Direct access to OpenAI's Responses API for multi-turn conversations with vision and audio capabilities.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/gateway\/openai\/v1\/responses/)).toBeInTheDocument();
  });

  test('renders provider selector in passthrough tab', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Passthrough APIs'));

    // Verify provider selector is rendered with all options
    expect(screen.getByRole('radio', { name: /OpenAI/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Anthropic/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Google Gemini/ })).toBeInTheDocument();

    // OpenAI is selected by default
    expect(screen.getByRole('radio', { name: /OpenAI/ })).toBeChecked();
  });

  test('renders language selector in passthrough tab', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Passthrough APIs'));

    // Verify language selector is rendered with both options
    expect(screen.getByRole('radio', { name: /cURL/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Python/ })).toBeInTheDocument();

    // cURL is selected by default and shows curl content
    expect(screen.getByRole('radio', { name: /cURL/ })).toBeChecked();
    expect(screen.getByText(/curl -X POST/)).toBeInTheDocument();
  });

  test('uses window.location.origin when baseUrl is not provided', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const originalLocation = window.location;
    Object.defineProperty(window, 'location', {
      value: { origin: 'http://custom-origin:8080' },
      writable: true,
    });

    renderWithDesignSystem(<EndpointUsageModal open onClose={jest.fn()} endpointName="my-endpoint" />);
    await userEvent.click(screen.getByText('Unified APIs'));

    // The URL should appear in multiple code examples, so use getAllByText
    const matchingElements = screen.getAllByText(/http:\/\/custom-origin:8080\/gateway\/my-endpoint/);
    expect(matchingElements.length).toBeGreaterThan(0);

    Object.defineProperty(window, 'location', { value: originalLocation, writable: true });
  });

  test('renders copy buttons for code examples', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByText('Unified APIs'));

    const copyButtons = screen.getAllByRole('button');
    expect(copyButtons.length).toBeGreaterThan(0);
  });

  describe('Try it tab', () => {
    const getRequestBodyTextarea = () => {
      const textareas = screen.getAllByRole('textbox');
      return textareas[0];
    };

    test('sends request and displays response on success', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;
      const mockResponse = { output: 'Hello from the model' };
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
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

    test('shows error when request body is invalid JSON', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      const requestBody = getRequestBodyTextarea();
      await userEvent.clear(requestBody);
      await userEvent.type(requestBody, 'not valid json');
      await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

      expect(fetchSpy).not.toHaveBeenCalled();
      expect(screen.getByText('Invalid JSON in request body')).toBeInTheDocument();
    });

    test('shows error and response body when request fails with HTTP error', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: () => Promise.resolve('Server error details'),
      });

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect(screen.getByText(/Request failed \(500\)/)).toBeInTheDocument();
      expect(screen.getByDisplayValue('Server error details')).toBeInTheDocument();
    });

    test('shows error when network request fails', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;
      fetchSpy.mockRejectedValueOnce(new Error('Network failure'));

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      await userEvent.click(screen.getByRole('button', { name: 'Send request' }));

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect(screen.getByText('Network failure')).toBeInTheDocument();
    });

    test('switching API type shows provider selector and updates request body for passthrough', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      expect(getRequestBodyTextarea().value).toContain('messages');

      await userEvent.click(screen.getByRole('radio', { name: 'Passthrough' }));
      expect(screen.getByText('Provider')).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: /OpenAI/ })).toBeInTheDocument();
      expect(getRequestBodyTextarea().value).toContain('model');
    });

    test('switching API type from passthrough back to unified resets request body to unified format', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      await userEvent.click(screen.getByRole('radio', { name: 'Passthrough' }));
      expect(getRequestBodyTextarea().value).toContain('model');

      await userEvent.click(screen.getByRole('radio', { name: 'Unified (MLflow Invocations)' }));
      expect(getRequestBodyTextarea().value).toContain('messages');
    });

    test('provider selection in Try it passthrough updates request body', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      await userEvent.click(screen.getByRole('radio', { name: 'Passthrough' }));

      await userEvent.click(screen.getByRole('radio', { name: 'Anthropic' }));
      expect(getRequestBodyTextarea().value).toContain('max_tokens');
      expect(getRequestBodyTextarea().value).toContain('messages');

      await userEvent.click(screen.getByRole('radio', { name: 'Google Gemini' }));
      expect(getRequestBodyTextarea().value).toContain('contents');
    });

    test('state reset when switching API type clears response and error', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ result: 'old' })),
      });

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
      expect(screen.getByText(/"result": "old"/)).toBeInTheDocument();

      await userEvent.click(screen.getByRole('radio', { name: 'Passthrough' }));
      expect(screen.queryByText(/"result": "old"/)).not.toBeInTheDocument();
      expect(screen.queryByText('Invalid JSON')).not.toBeInTheDocument();
    });

    test('state reset when switching provider clears response and error', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;
      fetchSpy.mockRejectedValueOnce(new Error('First error'));

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
      await userEvent.click(screen.getByRole('radio', { name: 'Passthrough' }));
      await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
      expect(screen.getByText('First error')).toBeInTheDocument();

      await userEvent.click(screen.getByRole('radio', { name: 'Anthropic' }));
      expect(screen.queryByText('First error')).not.toBeInTheDocument();
    });

    test('state reset when modal opens clears previous response and error', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ data: 'previous' })),
      });

      const { rerender } = renderWithDesignSystem(<EndpointUsageModal {...defaultProps} open />);
      await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
      expect(screen.getByText(/"data": "previous"/)).toBeInTheDocument();

      rerender(<EndpointUsageModal {...defaultProps} open={false} />);
      rerender(<EndpointUsageModal {...defaultProps} open />);

      expect(screen.queryByText(/"data": "previous"/)).not.toBeInTheDocument();
      expect(screen.getByText('Send request')).toBeInTheDocument();
    });

    test('Reset example restores default request body and clears response and error', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ out: 1 })),
      });

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
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
});
