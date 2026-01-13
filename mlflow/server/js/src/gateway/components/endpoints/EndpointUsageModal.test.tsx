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
      screen.getByText("Direct access to OpenAI's Responses API for multi-turn conversations with vision and audio capabilities."),
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
});
