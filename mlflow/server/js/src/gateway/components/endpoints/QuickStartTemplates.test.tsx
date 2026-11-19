import { describe, expect, it } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { MemoryRouter, Route, Routes, useLocation } from '../../../common/utils/RoutingUtils';
import { QuickStartTemplates, QuickStartTemplatesCompact } from './QuickStartTemplates';

const renderComponent = () =>
  renderWithDesignSystem(
    <MemoryRouter>
      <QuickStartTemplates />
    </MemoryRouter>,
  );

const renderCompactComponent = () =>
  renderWithDesignSystem(
    <MemoryRouter>
      <QuickStartTemplatesCompact />
    </MemoryRouter>,
  );

const LocationStateDisplay = () => {
  const location = useLocation();
  return <div data-testid="location-state">{JSON.stringify(location.state)}</div>;
};

const renderComponentWithCreateRoute = () =>
  renderWithDesignSystem(
    <MemoryRouter>
      <Routes>
        <Route path="/" element={<QuickStartTemplates />} />
        <Route path="/gateway/endpoints/create" element={<LocationStateDisplay />} />
      </Routes>
    </MemoryRouter>,
  );

describe('QuickStartTemplates', () => {
  it('renders the title and description', () => {
    renderComponent();

    expect(screen.getByText('Get started with AI Gateway')).toBeInTheDocument();
    expect(
      screen.getByText(
        'A Gateway endpoint routes your agent calls to any AI model, with built-in usage tracking through MLflow Tracing, budget controls, and more.',
      ),
    ).toBeInTheDocument();
  });

  it('renders all four provider cards', () => {
    renderComponent();

    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    expect(screen.getByText('Anthropic')).toBeInTheDocument();
    expect(screen.getByText('Google Gemini')).toBeInTheDocument();
    expect(screen.getByText('Databricks')).toBeInTheDocument();
  });

  it('renders model names on each card', () => {
    renderComponent();

    // OpenAI models
    expect(screen.getByText('gpt-5.6')).toBeInTheDocument();
    expect(screen.getByText('gpt-5.5')).toBeInTheDocument();
    expect(screen.getByText('gpt-5.4')).toBeInTheDocument();
    expect(screen.getByText('gpt-5.4-mini')).toBeInTheDocument();

    // Anthropic models
    expect(screen.getByText('claude-opus-5-5')).toBeInTheDocument();
    expect(screen.getByText('claude-sonnet-5-5')).toBeInTheDocument();
    expect(screen.getByText('claude-sonnet-4-6')).toBeInTheDocument();
    expect(screen.getByText('claude-haiku-5-5')).toBeInTheDocument();

    // Gemini models
    expect(screen.getByText('gemini-3.0-pro')).toBeInTheDocument();
    expect(screen.getByText('gemini-3.0-flash')).toBeInTheDocument();
    expect(screen.getByText('gemini-2.5-pro')).toBeInTheDocument();
    expect(screen.getByText('gemini-2.5-flash')).toBeInTheDocument();

    // Databricks models
    expect(screen.getByText('databricks-gpt-4.1')).toBeInTheDocument();
    expect(screen.getByText('databricks-claude-sonnet-4-6')).toBeInTheDocument();
    expect(screen.getByText('databricks-gemini-2.5-flash')).toBeInTheDocument();
    expect(screen.getByText('databricks-llama-4-maverick')).toBeInTheDocument();
  });

  it('renders cards as links to the create endpoint page with correct state', () => {
    renderComponent();

    const links = screen.getAllByRole('link');
    const cardLinks = links.filter((link) => link.getAttribute('href') === '/gateway/endpoints/create');
    // 4 providers × 4 models + 3 coding agents + 1 browse all link = 20
    expect(cardLinks.length).toBe(20);
  });

  it.each([
    ['openai', 'gpt-5.6', 'openai-gpt-5.6-endpoint', 'openai-api-key'],
    ['openai', 'gpt-5.5', 'openai-gpt-5.5-endpoint', 'openai-api-key'],
    ['openai', 'gpt-5.4', 'openai-gpt-5.4-endpoint', 'openai-api-key'],
    ['openai', 'gpt-5.4-mini', 'openai-gpt-5.4-mini-endpoint', 'openai-api-key'],
    ['anthropic', 'claude-opus-5-5', 'anthropic-claude-opus-endpoint', 'anthropic-api-key'],
    ['anthropic', 'claude-sonnet-5-5', 'anthropic-claude-sonnet-endpoint', 'anthropic-api-key'],
    ['anthropic', 'claude-sonnet-4-6', 'anthropic-claude-sonnet-4-6-endpoint', 'anthropic-api-key'],
    ['anthropic', 'claude-haiku-5-5', 'anthropic-claude-haiku-endpoint', 'anthropic-api-key'],
  ])('passes %s model %s and endpoint name %s in the link state', async (provider, model, endpointName, secretName) => {
    renderComponentWithCreateRoute();

    await userEvent.click(screen.getByText(model));

    expect(JSON.parse(screen.getByTestId('location-state').textContent ?? 'null')).toEqual({
      provider,
      model,
      endpointName,
      secretName,
    });
  });

  it('renders coding agents section with links to documentation', () => {
    renderComponent();

    expect(screen.getByText('Coding Agents')).toBeInTheDocument();
    expect(screen.getByText('Claude Code')).toBeInTheDocument();
    expect(screen.getByText('OpenAI Codex')).toBeInTheDocument();
    expect(screen.getByText('Gemini CLI')).toBeInTheDocument();
  });

  it('renders the browse all providers link', () => {
    renderComponent();

    expect(screen.getByText('Or browse all providers and models →')).toBeInTheDocument();
  });
});

describe('QuickStartTemplatesCompact', () => {
  it('renders all four provider cards with models', () => {
    renderCompactComponent();

    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    expect(screen.getByText('Anthropic')).toBeInTheDocument();
    expect(screen.getByText('Google Gemini')).toBeInTheDocument();
    expect(screen.getByText('Databricks')).toBeInTheDocument();
  });

  it('renders the quick start label', () => {
    renderCompactComponent();

    expect(screen.getByText('Quick start')).toBeInTheDocument();
  });

  it('renders the browse all providers link', () => {
    renderCompactComponent();

    expect(screen.getByText('Browse all providers →')).toBeInTheDocument();
  });

  it('renders model links for all providers', () => {
    renderCompactComponent();

    const links = screen.getAllByRole('link');
    const cardLinks = links.filter((link) => link.getAttribute('href') === '/gateway/endpoints/create');
    // 4 providers × 4 models + 3 coding agents + 1 browse all link = 20
    expect(cardLinks.length).toBe(20);
  });

  it('renders coding agents section with links to documentation', () => {
    renderCompactComponent();

    expect(screen.getByText('Coding Agents')).toBeInTheDocument();
    expect(screen.getByText('Claude Code')).toBeInTheDocument();
    expect(screen.getByText('OpenAI Codex')).toBeInTheDocument();
    expect(screen.getByText('Gemini CLI')).toBeInTheDocument();
  });
});
