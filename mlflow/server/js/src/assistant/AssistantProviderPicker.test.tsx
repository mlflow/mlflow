import { beforeEach, describe, test, expect, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { AssistantProviderPicker } from './AssistantProviderPicker';
import type { AssistantProviderSelection, ProviderInfo, ResolvedProviderInfo } from './types';
import type { Endpoint } from '../gateway/types';

let mockGatewayEndpoints: Endpoint[] = [];

jest.mock('../gateway/hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: () => ({
    data: mockGatewayEndpoints,
    isLoading: false,
  }),
}));

const providerInfo = (overrides: Partial<ProviderInfo> & { name: string; display_name?: string }): ProviderInfo => ({
  display_name: overrides.display_name ?? overrides.name,
  description: '',
  available: true,
  selected: false,
  requires_api_key: false,
  has_api_key: false,
  allows_remote_access: false,
  model_options: [],
  ...overrides,
});

const resolvedProvider = (overrides: Partial<ResolvedProviderInfo> = {}): ResolvedProviderInfo => ({
  name: 'claude_code',
  model: null,
  auto_selected: true,
  requires_api_key: false,
  has_api_key: false,
  ...overrides,
});

const endpoint = (name: string, modelName = 'model-a'): Endpoint =>
  ({
    endpoint_id: name,
    name,
    model_mappings: [
      {
        mapping_id: `${name}-mapping`,
        endpoint_id: name,
        model_definition_id: `${name}-model-definition`,
        model_definition: { model_name: modelName },
        weight: 1,
        created_at: 0,
      },
    ],
    created_at: 0,
    last_updated_at: 0,
  }) as Endpoint;

const renderPicker = ({
  provider = resolvedProvider(),
  providers = [
    providerInfo({ name: 'claude_code', display_name: 'Claude Code' }),
    providerInfo({ name: 'codex', display_name: 'OpenAI Codex' }),
    providerInfo({ name: 'mlflow_gateway', display_name: 'MLflow AI Gateway' }),
  ],
  gatewayVendorOptions = { openai: ['gpt-5.5', 'gpt-5-mini'] },
  onSelect = jest.fn(),
}: {
  provider?: ResolvedProviderInfo;
  providers?: ProviderInfo[];
  gatewayVendorOptions?: Record<string, string[]>;
  onSelect?: (selection: AssistantProviderSelection) => void;
} = {}) => {
  const result = renderWithIntl(
    <DesignSystemProvider>
      <AssistantProviderPicker
        provider={provider}
        providers={providers}
        gatewayVendorOptions={gatewayVendorOptions}
        onSelect={onSelect}
      />
    </DesignSystemProvider>,
  );
  return { ...result, onSelect };
};

describe('AssistantProviderPicker', () => {
  beforeEach(() => {
    mockGatewayEndpoints = [];
  });

  test('groups API provider shortcuts separately from local providers and gateway endpoints', async () => {
    const user = userEvent.setup();
    const { onSelect } = renderPicker({
      gatewayVendorOptions: { openai: ['gpt-5.5', 'gpt-5-mini'], anthropic: ['claude-3-5-sonnet'] },
    });

    await user.click(screen.getByRole('button', { name: 'Change assistant provider' }));

    expect(await screen.findByText('API providers')).toBeInTheDocument();
    expect(screen.getByText('OpenAI API')).toBeInTheDocument();
    expect(screen.getByText('Anthropic API')).toBeInTheDocument();
    expect(screen.getByText('Local providers')).toBeInTheDocument();
    expect(screen.getByText('Codex CLI (local)')).toBeInTheDocument();
    expect(screen.getByText('Gateway endpoints')).toBeInTheDocument();
    expect(screen.getByText('MLflow AI Gateway')).toBeInTheDocument();
    expect(screen.queryByText('OpenAI Codex')).not.toBeInTheDocument();

    await user.click(screen.getByText('OpenAI API'));
    expect(onSelect).toHaveBeenCalledWith({
      kind: 'gateway',
      endpointName: 'mlflow-assistant-openai',
      gatewayVendor: 'openai',
      providerModel: 'gpt-5.5',
      modelOptions: ['gpt-5.5', 'gpt-5-mini'],
      requiresApiKey: true,
      hasApiKey: false,
    });
  });

  test('hides assistant-managed gateway endpoints from the custom gateway submenu', async () => {
    const user = userEvent.setup();
    mockGatewayEndpoints = [
      endpoint('mlflow-assistant-anthropic', 'claude-3-5-sonnet'),
      endpoint('custom-anthropic-endpoint', 'claude-3-opus'),
    ];
    renderPicker({ gatewayVendorOptions: { anthropic: ['claude-3-5-sonnet'] } });

    await user.click(screen.getByRole('button', { name: 'Change assistant provider' }));
    expect(await screen.findByText('Anthropic API')).toBeInTheDocument();

    await user.hover(screen.getByText('MLflow AI Gateway'));
    expect(await screen.findByText('custom-anthropic-endpoint')).toBeInTheDocument();
    expect(screen.queryByText('mlflow-assistant-anthropic')).not.toBeInTheDocument();
  });

  test('switches the current hosted model optimistically', async () => {
    const user = userEvent.setup();
    const { onSelect } = renderPicker({
      provider: resolvedProvider({
        name: 'mlflow_gateway',
        model: 'mlflow-assistant-openai',
        auto_selected: false,
        model_provider: 'openai',
        provider_model: 'gpt-5.5',
        model_options: ['gpt-5.5', 'gpt-5-mini'],
        requires_api_key: true,
        has_api_key: false,
      }),
    });

    await user.click(screen.getByRole('button', { name: 'Change assistant model' }));
    await user.click(await screen.findByText('gpt-5-mini'));

    expect(onSelect).toHaveBeenCalledWith({
      kind: 'gateway',
      endpointName: 'mlflow-assistant-openai',
      gatewayVendor: 'openai',
      providerModel: 'gpt-5-mini',
      modelOptions: ['gpt-5.5', 'gpt-5-mini'],
      requiresApiKey: true,
      hasApiKey: false,
    });
  });

  test.each([
    ['openai', 'OpenAI API'],
    ['xai', 'MLflow AI Gateway'],
  ])('shows %s gateway endpoint as %s', (modelProvider, label) => {
    renderPicker({
      provider: resolvedProvider({
        name: 'mlflow_gateway',
        model: 'chat-endpoint',
        model_provider: modelProvider,
      }),
    });

    expect(screen.getByText(label)).toBeInTheDocument();
  });
});
