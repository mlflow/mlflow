import { describe, expect, it } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { StarterCodeCard } from './StarterCodeCard';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

const renderCard = (props: { endpointName: string; provider?: string }) =>
  renderWithDesignSystem(
    <MemoryRouter>
      <StarterCodeCard {...props} />
    </MemoryRouter>,
  );

describe('StarterCodeCard', () => {
  it('renders with MLflow Chat Completions as default', () => {
    renderCard({ endpointName: 'my-endpoint' });
    expect(screen.getByText('View starter code')).toBeInTheDocument();
    expect(screen.getByText('MLflow Chat Completions')).toBeInTheDocument();
    expect(screen.getByText('Try in Browser')).toBeInTheDocument();
  });

  it('shows cURL tab as default with endpoint name in code', () => {
    renderCard({ endpointName: 'my-endpoint' });
    // cURL tab should be active by default and code should contain the endpoint name
    const codeBlock = document.querySelector('pre');
    expect(codeBlock?.textContent).toContain('my-endpoint');
    expect(codeBlock?.textContent).toContain('chat/completions');
  });

  it('shows only MLflow Chat Completions when provider has no passthrough', () => {
    renderCard({ endpointName: 'my-endpoint', provider: 'cohere' });
    expect(screen.getByText('MLflow Chat Completions')).toBeInTheDocument();
    expect(screen.queryByText('OpenAI Responses')).not.toBeInTheDocument();
    expect(screen.queryByText('Anthropic Messages')).not.toBeInTheDocument();
    expect(screen.queryByText('Gemini Generate Content')).not.toBeInTheDocument();
  });

  it('shows OpenAI Responses tab for openai provider', () => {
    renderCard({ endpointName: 'my-endpoint', provider: 'openai' });
    expect(screen.getByText('MLflow Chat Completions')).toBeInTheDocument();
    expect(screen.getByText('OpenAI Responses')).toBeInTheDocument();
  });

  it('shows OpenAI Responses tab for azure provider', () => {
    renderCard({ endpointName: 'my-endpoint', provider: 'azure' });
    expect(screen.getByText('OpenAI Responses')).toBeInTheDocument();
  });

  it('shows Anthropic Messages tab for anthropic provider', () => {
    renderCard({ endpointName: 'my-endpoint', provider: 'anthropic' });
    expect(screen.getByText('Anthropic Messages')).toBeInTheDocument();
  });

  it('shows Gemini Generate Content tab for gemini provider', () => {
    renderCard({ endpointName: 'my-endpoint', provider: 'gemini' });
    expect(screen.getByText('Gemini Generate Content')).toBeInTheDocument();
  });

  it('switches to Python code when Python tab is clicked', async () => {
    renderCard({ endpointName: 'my-endpoint' });
    await userEvent.click(screen.getByText('Python'));
    const codeBlock = document.querySelector('pre');
    expect(codeBlock?.textContent).toContain('from openai import OpenAI');
  });

  it('switches API variant when segmented control is clicked', async () => {
    renderCard({ endpointName: 'my-endpoint', provider: 'anthropic' });
    await userEvent.click(screen.getByText('Anthropic Messages'));
    const codeBlock = document.querySelector('pre');
    expect(codeBlock?.textContent).toContain('anthropic/v1/messages');
  });

  it('includes unified comment in chat-completions code', () => {
    renderCard({ endpointName: 'my-endpoint' });
    const codeBlock = document.querySelector('pre');
    expect(codeBlock?.textContent).toContain('Unified OpenAI compatible API');
  });

  it('includes provider description comment in passthrough code', async () => {
    renderCard({ endpointName: 'my-endpoint', provider: 'openai' });
    await userEvent.click(screen.getByText('OpenAI Responses'));
    const codeBlock = document.querySelector('pre');
    expect(codeBlock?.textContent).toContain("Passthrough to OpenAI's Responses API");
    expect(codeBlock?.textContent).toContain('New OpenAI features are available immediately');
  });
});
