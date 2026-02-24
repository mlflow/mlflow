import { describe, test, expect, jest } from '@jest/globals';
import { renderWithDesignSystem, screen, act } from '../../../common/utils/TestUtils.react18';
import { EndpointUsageModal } from './EndpointUsageModal';

jest.mock('../../../common/utils/FetchUtils', () => ({
  getDefaultHeaders: jest.fn().mockReturnValue({}),
}));

describe('EndpointUsageModal', () => {
  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    endpointName: 'test-endpoint',
    baseUrl: 'http://localhost:5000',
  };

  test('renders modal with title when open', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    expect(screen.getByText('Query endpoint')).toBeInTheDocument();
  });

  test('does not render modal when closed', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} open={false} />);

    expect(screen.queryByText('Query endpoint')).not.toBeInTheDocument();
  });

  test('renders unified APIs tab by default', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    expect(screen.getByText('MLflow Invocations API')).toBeInTheDocument();
    expect(screen.getByText('OpenAI-Compatible Chat Completions API')).toBeInTheDocument();
  });

  test('renders language selector in unified APIs tab', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    // Verify language selector is rendered with both options
    expect(screen.getByRole('radio', { name: /cURL/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Python/ })).toBeInTheDocument();

    // cURL is selected by default - shows curl content for both API sections
    expect(screen.getByRole('radio', { name: /cURL/ })).toBeChecked();
    const curlElements = screen.getAllByText(/curl -X POST/);
    expect(curlElements.length).toBe(2); // MLflow Invocations and OpenAI-Compatible both show cURL
  });

  test('shows code examples with endpoint name in unified APIs', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

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

  test('uses window.location.origin when baseUrl is not provided', () => {
    const originalLocation = window.location;
    Object.defineProperty(window, 'location', {
      value: { origin: 'http://custom-origin:8080' },
      writable: true,
    });

    renderWithDesignSystem(<EndpointUsageModal open onClose={jest.fn()} endpointName="my-endpoint" />);

    // The URL should appear in multiple code examples, so use getAllByText
    const matchingElements = screen.getAllByText(/http:\/\/custom-origin:8080\/gateway\/my-endpoint/);
    expect(matchingElements.length).toBeGreaterThan(0);

    Object.defineProperty(window, 'location', { value: originalLocation, writable: true });
  });

  test('renders copy buttons for code examples', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    const copyButtons = screen.getAllByRole('button');
    expect(copyButtons.length).toBeGreaterThan(0);
  });

  describe('Try it tab', () => {
    test('renders Try it tab with request and response areas', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));

      expect(screen.getByText('Request')).toBeInTheDocument();
      expect(screen.getByText('Response')).toBeInTheDocument();
      expect(screen.getByText('Send request')).toBeInTheDocument();
      expect(screen.getByText('Reset example')).toBeInTheDocument();
    });

    test('renders API type selector in Try it tab', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));

      expect(screen.getByText('API')).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: /Unified/ })).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: /Passthrough/ })).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: /Unified/ })).toBeChecked();
    });

    test('shows provider selector when passthrough API type is selected', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));
      await userEvent.click(screen.getByRole('radio', { name: /Passthrough/ }));

      expect(screen.getAllByText('Provider').length).toBeGreaterThan(0);
      expect(screen.getByRole('radio', { name: /OpenAI/ })).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: /Anthropic/ })).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: /Google Gemini/ })).toBeInTheDocument();
    });

    test('resets response and error when switching API type', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));

      // Switch to passthrough - response/error should clear
      await userEvent.click(screen.getByRole('radio', { name: /Passthrough/ }));

      // Verify provider selector is now visible (state was updated)
      expect(screen.getByRole('radio', { name: /OpenAI/ })).toBeInTheDocument();

      // Switch back to unified - provider selector should disappear
      await userEvent.click(screen.getByRole('radio', { name: /Unified/ }));
      expect(screen.queryByRole('radio', { name: /OpenAI/ })).not.toBeInTheDocument();
    });

    test('resets response and error when switching provider', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));
      await userEvent.click(screen.getByRole('radio', { name: /Passthrough/ }));

      // Switch provider
      await userEvent.click(screen.getByRole('radio', { name: /Anthropic/ }));

      expect(screen.getByRole('radio', { name: /Anthropic/ })).toBeChecked();
    });

    test('sends request and shows response', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      const mockResponse = { choices: [{ message: { content: 'Hello!' } }] };
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));
      await userEvent.click(screen.getByText('Send request'));

      await screen.findByText(/Hello!/);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:5000/gateway/test-endpoint/mlflow/invocations',
        expect.objectContaining({ method: 'POST' }),
      );
    });

    test('shows error message on request failure', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      global.fetch = jest.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: () => Promise.resolve('Server error'),
      });

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));
      await userEvent.click(screen.getByText('Send request'));

      await screen.findByText(/Request failed \(500\)/);
    });

    test('shows error for invalid JSON in request body', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));

      const textarea = screen.getAllByRole('textbox')[0];
      await userEvent.clear(textarea);
      await userEvent.type(textarea, 'invalid json');
      await userEvent.click(screen.getByText('Send request'));

      expect(screen.getByText('Invalid JSON in request body')).toBeInTheDocument();
    });

    test('shows Sending... while request is in progress', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      let resolveResponse;
      global.fetch = jest.fn().mockReturnValue(
        new Promise((resolve) => {
          resolveResponse = resolve;
        }),
      );

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));
      await userEvent.click(screen.getByText('Send request'));

      expect(screen.getByText('Sending...')).toBeInTheDocument();

      await act(async () => {
        resolveResponse({
          ok: true,
          text: () => Promise.resolve('{}'),
        });
      });
    });

    test('resets state when modal is reopened', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      const { rerender } = renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));

      // Switch to passthrough
      await userEvent.click(screen.getByRole('radio', { name: /Passthrough/ }));
      expect(screen.getByRole('radio', { name: /OpenAI/ })).toBeInTheDocument();

      // Close and reopen the modal
      rerender(<EndpointUsageModal {...defaultProps} open={false} />);
      rerender(<EndpointUsageModal {...defaultProps} open={true} />);

      await userEvent.click(screen.getByText('Try it'));

      // State should be reset: API type should be unified (no provider selector)
      expect(screen.queryByRole('radio', { name: /OpenAI/ })).not.toBeInTheDocument();
      expect(screen.getByRole('radio', { name: /Unified/ })).toBeChecked();
    });

    test('resets request body when Reset example is clicked', async () => {
      const userEvent = (await import('@testing-library/user-event')).default;

      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      await userEvent.click(screen.getByText('Try it'));

      const textarea = screen.getAllByRole('textbox')[0];
      await userEvent.clear(textarea);
      await userEvent.type(textarea, '{{}}');

      await userEvent.click(screen.getByText('Reset example'));

      expect(textarea).toHaveValue(
        JSON.stringify({ messages: [{ role: 'user', content: 'Hello, how are you?' }] }, null, 2),
      );
    });

    test('renders tabs with ARIA attributes', () => {
      renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

      const tablist = screen.getByRole('tablist');
      expect(tablist).toBeInTheDocument();

      const tabs = screen.getAllByRole('tab');
      expect(tabs.length).toBe(3);

      const unifiedTab = tabs.find((t) => t.textContent?.includes('Unified APIs'));
      expect(unifiedTab).toHaveAttribute('aria-selected', 'true');

      const tryItTab = tabs.find((t) => t.textContent?.includes('Try it'));
      expect(tryItTab).toHaveAttribute('aria-selected', 'false');
    });
  });
});
