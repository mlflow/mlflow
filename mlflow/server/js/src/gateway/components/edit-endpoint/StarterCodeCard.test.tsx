import { afterEach, describe, expect, it, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { StarterCodeCard } from './StarterCodeCard';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import * as FetchUtils from '../../../common/utils/FetchUtils';

const renderCard = (props: { endpointName: string; provider?: string }) =>
  renderWithDesignSystem(
    <MemoryRouter>
      <QueryClientProvider client={new QueryClient()}>
        <StarterCodeCard {...props} />
      </QueryClientProvider>
    </MemoryRouter>,
  );

describe('StarterCodeCard', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

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

  it('shows only the System One API for TypeSafe endpoints', () => {
    renderCard({ endpointName: 'jev-evaluator', provider: 'typesafe' });

    expect(screen.getByText('TypeSafe System One')).toBeInTheDocument();
    expect(screen.queryByText('MLflow Chat Completions')).not.toBeInTheDocument();
    const code = document.querySelector('pre')?.textContent;
    expect(code).toContain('/gateway/typesafe/v1/systemone');
    expect(code).toContain('"model": "jev-evaluator"');
    expect(code).toContain('"type": "noul"');
    expect(code).not.toContain('messages');
  });

  it('updates the API when the endpoint provider changes', () => {
    const { rerender } = renderWithDesignSystem(<StarterCodeCard endpointName="my-endpoint" provider="openai" />);
    rerender(<StarterCodeCard endpointName="my-endpoint" provider="typesafe" />);

    expect(screen.queryByText('MLflow Chat Completions')).not.toBeInTheDocument();
    expect(document.querySelector('pre')?.textContent).toContain('/gateway/typesafe/v1/systemone');
  });

  it('shows Python code that reads the typed answer', async () => {
    renderCard({ endpointName: 'jev-evaluator', provider: 'typesafe' });
    await userEvent.click(screen.getByText('Python'));

    const code = document.querySelector('pre')?.textContent;
    expect(code).toContain('import requests');
    expect(code).toContain('/gateway/typesafe/v1/systemone');
    expect(code).toContain('response.raise_for_status()');
    expect(code).toContain('response.json()["answers"]["evaluation"]');
  });

  it('sends a typed System One request and displays the probability', async () => {
    const response = { answers: { evaluation: { type: 'noul', noul: 0.95 } } };
    const fetchSpy = jest.spyOn(FetchUtils, 'fetchOrFail').mockResolvedValueOnce({
      text: () => Promise.resolve(JSON.stringify(response)),
    } as Response);
    renderCard({ endpointName: 'jev-evaluator', provider: 'typesafe' });
    await userEvent.click(screen.getByText('Try in Browser'));

    const request = JSON.parse((screen.getAllByRole('textbox')[0] as HTMLTextAreaElement).value);
    expect(request).toEqual({
      model: 'jev-evaluator',
      state: { inputs: { question: 'What is the capital of France?' }, outputs: 'Paris.' },
      questions: {
        evaluation: {
          type: 'noul',
          instructions: 'Does the answer correctly address the question?',
          criteria: { true: 'The answer is correct and relevant.', false: 'The answer is incorrect or irrelevant.' },
        },
      },
    });

    await userEvent.click(screen.getByRole('button', { name: 'Send request' }));
    expect(fetchSpy).toHaveBeenCalledWith(`${window.location.origin}/gateway/typesafe/v1/systemone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    expect(await screen.findByDisplayValue(/"noul": 0.95/)).toHaveValue(JSON.stringify(response, null, 2));
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
