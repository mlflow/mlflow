import { describe, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { ProviderCell } from './ProviderCell';
import type { Endpoint } from '../../types';

const createModelMapping = (provider: string, modelName: string, mappingId: string): Endpoint['model_mappings'][0] => ({
  mapping_id: mappingId,
  endpoint_id: 'ep-123',
  model_definition_id: `md-${mappingId}`,
  model_definition: {
    model_definition_id: `md-${mappingId}`,
    name: `model-${mappingId}`,
    secret_id: 's-123',
    secret_name: 'test-secret',
    provider,
    model_name: modelName,
    endpoint_count: 1,
    created_at: Date.now() / 1000,
    last_updated_at: Date.now() / 1000,
  },
  weight: 1,
  created_at: Date.now() / 1000,
});

describe('ProviderCell', () => {
  test('renders dash when no model mappings', () => {
    renderWithDesignSystem(<ProviderCell modelMappings={[]} />);
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders dash when model mappings is undefined', () => {
    renderWithDesignSystem(<ProviderCell modelMappings={undefined as any} />);
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders single provider', () => {
    const modelMappings = [createModelMapping('openai', 'gpt-4', '1')];
    renderWithDesignSystem(<ProviderCell modelMappings={modelMappings} />);
    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    expect(screen.queryByText(/more/)).not.toBeInTheDocument();
  });

  test('shows primary provider with "+N more" link and expands on click', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const modelMappings = [
      createModelMapping('openai', 'gpt-4', '1'),
      createModelMapping('anthropic', 'claude-3', '2'),
      createModelMapping('gemini', 'gemini-pro', '3'),
    ];
    renderWithDesignSystem(<ProviderCell modelMappings={modelMappings} />);

    // Initially shows only primary provider
    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    expect(screen.queryByText('Anthropic')).not.toBeInTheDocument();
    expect(screen.queryByText('Google Gemini')).not.toBeInTheDocument();
    expect(screen.getByText('+2 more')).toBeInTheDocument();

    // Click to expand
    await userEvent.click(screen.getByText('+2 more'));

    // All providers should now be visible
    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    expect(screen.getByText('Anthropic')).toBeInTheDocument();
    expect(screen.getByText('Google Gemini')).toBeInTheDocument();
    expect(screen.getByText('Show less')).toBeInTheDocument();

    // Click to collapse
    await userEvent.click(screen.getByText('Show less'));

    // Back to showing only primary provider
    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    expect(screen.queryByText('Anthropic')).not.toBeInTheDocument();
    expect(screen.getByText('+2 more')).toBeInTheDocument();
  });

  test('deduplicates providers when same provider appears multiple times', () => {
    const modelMappings = [
      createModelMapping('openai', 'gpt-4', '1'),
      createModelMapping('openai', 'gpt-3.5-turbo', '2'),
      createModelMapping('anthropic', 'claude-3', '3'),
    ];
    renderWithDesignSystem(<ProviderCell modelMappings={modelMappings} />);

    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    // Only +1 more because OpenAI is deduplicated
    expect(screen.getByText('+1 more')).toBeInTheDocument();
  });

  test('renders dash when model definition has no provider', () => {
    const modelMappings = [
      {
        mapping_id: '1',
        endpoint_id: 'ep-123',
        model_definition_id: 'md-1',
        model_definition: {
          model_definition_id: 'md-1',
          name: 'test-model',
          secret_id: 's-123',
          secret_name: 'test-secret',
          provider: undefined as any,
          model_name: 'test',
          endpoint_count: 1,
          created_at: Date.now() / 1000,
          last_updated_at: Date.now() / 1000,
        },
        weight: 1,
        created_at: Date.now() / 1000,
      },
    ];
    renderWithDesignSystem(<ProviderCell modelMappings={modelMappings} />);
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('formats provider names correctly', () => {
    const modelMappings = [createModelMapping('bedrock', 'claude-v2', '1')];
    renderWithDesignSystem(<ProviderCell modelMappings={modelMappings} />);
    expect(screen.getByText('Amazon Bedrock')).toBeInTheDocument();
  });
});
