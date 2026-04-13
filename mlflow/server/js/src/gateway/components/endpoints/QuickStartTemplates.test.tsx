import { describe, expect, it } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { QuickStartTemplates } from './QuickStartTemplates';

const renderComponent = () =>
  renderWithDesignSystem(
    <MemoryRouter>
      <QuickStartTemplates />
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

    expect(screen.getByText('gpt-5.4')).toBeInTheDocument();
    expect(screen.getByText('claude-sonnet-4-6')).toBeInTheDocument();
    expect(screen.getByText('gemini-2.5-pro')).toBeInTheDocument();
    expect(screen.getByText('databricks-gpt-5')).toBeInTheDocument();
  });

  it('renders cards as links to the create endpoint page with correct state', () => {
    renderComponent();

    const links = screen.getAllByRole('link');
    const cardLinks = links.filter((link) => link.getAttribute('href') === '/gateway/endpoints/create');
    // 4 provider cards + 1 browse all link = 5
    expect(cardLinks.length).toBe(5);
  });

  it('renders the browse all providers link', () => {
    renderComponent();

    expect(screen.getByText('Or browse all providers and models →')).toBeInTheDocument();
  });
});
