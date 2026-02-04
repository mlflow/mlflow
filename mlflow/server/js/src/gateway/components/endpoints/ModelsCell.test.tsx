import { describe, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { ModelsCell } from './ModelsCell';
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

describe('ModelsCell', () => {
  test('renders dash when no model mappings', () => {
    renderWithDesignSystem(<ModelsCell modelMappings={[]} />);
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders dash when model mappings is undefined', () => {
    renderWithDesignSystem(<ModelsCell modelMappings={undefined as any} />);
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders single model without expand link', () => {
    const modelMappings = [createModelMapping('openai', 'gpt-4', '1')];
    renderWithDesignSystem(<ModelsCell modelMappings={modelMappings} />);
    expect(screen.getByText('gpt-4')).toBeInTheDocument();
    expect(screen.queryByText(/more/)).not.toBeInTheDocument();
  });

  test('shows primary model with "+N more" link and expands on click', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const modelMappings = [
      createModelMapping('openai', 'gpt-4', '1'),
      createModelMapping('anthropic', 'claude-3-opus', '2'),
      createModelMapping('gemini', 'gemini-pro', '3'),
    ];
    renderWithDesignSystem(<ModelsCell modelMappings={modelMappings} />);

    // Initially shows only primary model
    expect(screen.getByText('gpt-4')).toBeInTheDocument();
    expect(screen.queryByText('claude-3-opus')).not.toBeInTheDocument();
    expect(screen.queryByText('gemini-pro')).not.toBeInTheDocument();
    expect(screen.getByText('+2 more')).toBeInTheDocument();

    // Click to expand
    await userEvent.click(screen.getByText('+2 more'));

    // All models should now be visible
    expect(screen.getByText('gpt-4')).toBeInTheDocument();
    expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    expect(screen.getByText('gemini-pro')).toBeInTheDocument();
    expect(screen.getByText('Show less')).toBeInTheDocument();

    // Click to collapse
    await userEvent.click(screen.getByText('Show less'));

    // Back to showing only primary model
    expect(screen.getByText('gpt-4')).toBeInTheDocument();
    expect(screen.queryByText('claude-3-opus')).not.toBeInTheDocument();
    expect(screen.getByText('+2 more')).toBeInTheDocument();
  });

  test('renders dash for model with missing model_name', () => {
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
          provider: 'openai',
          model_name: undefined as any,
          endpoint_count: 1,
          created_at: Date.now() / 1000,
          last_updated_at: Date.now() / 1000,
        },
        weight: 1,
        created_at: Date.now() / 1000,
      },
    ];
    renderWithDesignSystem(<ModelsCell modelMappings={modelMappings} />);
    expect(screen.getByText('-')).toBeInTheDocument();
  });
});
