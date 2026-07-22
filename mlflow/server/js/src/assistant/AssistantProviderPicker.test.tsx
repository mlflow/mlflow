import { describe, test, expect, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { AssistantProviderPicker } from './AssistantProviderPicker';
import type { AssistantProviderSelection, ProviderInfo, ResolvedProviderInfo } from './types';

jest.mock('../gateway/hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: () => ({
    data: [],
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
  test('lists hosted gateway vendors as top-level shortcuts', async () => {
    const user = userEvent.setup();
    const { onSelect } = renderPicker();

    await user.click(screen.getByRole('button', { name: 'Change assistant provider' }));

    expect(await screen.findByText('OpenAI')).toBeInTheDocument();
    expect(screen.getByText('MLflow AI Gateway')).toBeInTheDocument();
    expect(screen.getByText('Codex')).toBeInTheDocument();
    expect(screen.queryByText('OpenAI Codex')).not.toBeInTheDocument();

    await user.click(screen.getByText('OpenAI'));
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
    ['openai', 'OpenAI'],
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
