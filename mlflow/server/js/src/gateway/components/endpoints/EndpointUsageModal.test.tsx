import { describe, test, expect, jest } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { EndpointUsageModal } from './EndpointUsageModal';

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

  test('renders Try it by default in Unified APIs tab', () => {
    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);

    expect(screen.getByText('Send request')).toBeInTheDocument();
    expect(screen.getByText('Request')).toBeInTheDocument();
    expect(screen.getByText('Response')).toBeInTheDocument();
  });

  test('Unified APIs tab shows Try it, cURL, and Python view options', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    // Unified APIs is selected by default; view mode selector has Try it | cURL | Python
    expect(screen.getByRole('radio', { name: /cURL/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Python/ })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'cURL' }));
    const curlElements = screen.getAllByText(/curl -X POST/);
    expect(curlElements.length).toBe(1); // Only the selected unified variant (MLflow Invocations by default) is shown
  });

  test('shows code examples with endpoint name in unified APIs when cURL selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByRole('radio', { name: 'cURL' }));

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

  test('shows OpenAI passthrough example in passthrough tab when cURL selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByText('Passthrough APIs'));
    await userEvent.click(screen.getByRole('radio', { name: 'cURL' }));

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

  test('passthrough tab shows Try it, cURL, Python and code when cURL selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByText('Passthrough APIs'));

    expect(screen.getByRole('radio', { name: /cURL/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Python/ })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'cURL' }));
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
    await userEvent.click(screen.getByRole('radio', { name: 'cURL' }));

    const matchingElements = screen.getAllByText(/http:\/\/custom-origin:8080\/gateway\/my-endpoint/);
    expect(matchingElements.length).toBeGreaterThan(0);

    Object.defineProperty(window, 'location', { value: originalLocation, writable: true });
  });

  test('renders copy buttons for code examples when cURL selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<EndpointUsageModal {...defaultProps} />);
    await userEvent.click(screen.getByRole('radio', { name: 'cURL' }));

    const copyButtons = screen.getAllByRole('button');
    expect(copyButtons.length).toBeGreaterThan(0);
  });
});
