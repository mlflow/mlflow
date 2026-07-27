import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModelDropdown, type IssueDetectionModelSelection } from './IssueDetectionModelDropdown';
import { useModelsQuery } from '../../../../../gateway/hooks/useModelsQuery';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { useSecretsQuery } from '../../../../../gateway/hooks/useSecretsQuery';

jest.mock('../../../../../gateway/hooks/useModelsQuery', () => ({
  useModelsQuery: jest.fn(),
}));
jest.mock('../../../../../gateway/hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: jest.fn(),
}));
jest.mock('../../../../../gateway/hooks/useSecretsQuery', () => ({
  useSecretsQuery: jest.fn(),
}));
jest.mock('../../../../../gateway/components/endpoint-form', () => ({
  CreateEndpointModal: ({ open, onSuccess }: { open: boolean; onSuccess: (e: { name: string }) => void }) =>
    open ? (
      <div data-testid="create-endpoint-modal">
        <button data-testid="create-endpoint-submit" onClick={() => onSuccess({ name: 'new-endpoint' })}>
          create
        </button>
      </div>
    ) : null,
}));

const OPENAI_SELECTION: IssueDetectionModelSelection = {
  mode: 'direct',
  provider: 'openai',
  model: 'gpt-5.6-sol',
};

describe('IssueDetectionModelDropdown', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useModelsQuery).mockImplementation(
      ({ provider } = {}) =>
        ({
          data:
            provider === 'anthropic'
              ? [{ model: 'claude-sonnet-4-5' }, { model: 'claude-opus-4-8' }, { model: 'claude-sonnet-4-6' }]
              : undefined,
          error: undefined,
          isLoading: false,
          refetch: jest.fn(),
        }) as any,
    );
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [], isLoading: false, refetch: jest.fn() } as any);
    jest.mocked(useSecretsQuery).mockReturnValue({ data: [], isLoading: false, refetch: jest.fn() } as any);
  });

  const openDropdown = async () => userEvent.click(screen.getByTestId('model-dropdown-trigger'));

  test('shows the current selection on the trigger card', () => {
    renderWithDesignSystem(
      <IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={jest.fn()} />,
    );

    const trigger = screen.getByTestId('model-dropdown-trigger');
    expect(trigger).toHaveTextContent('OpenAI');
    expect(trigger).toHaveTextContent('gpt-5.6-sol');
  });

  test('AI Gateway is always a group, even with no endpoints, offering to create one', async () => {
    const onChange = jest.fn();
    renderWithDesignSystem(<IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={onChange} />);

    await openDropdown();
    expect(screen.getByTestId('model-group-gateway')).toBeInTheDocument();
    expect(screen.queryByTestId('model-create-endpoint')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('model-group-gateway'));
    expect(screen.getByTestId('model-create-endpoint')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('model-create-endpoint'));
    await userEvent.click(screen.getByTestId('create-endpoint-submit'));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ mode: 'endpoint', endpointName: 'new-endpoint' }));
  });

  test('lists AI Gateway endpoints inside the gateway group and selects one', async () => {
    const onChange = jest.fn();
    renderWithDesignSystem(
      <IssueDetectionModelDropdown
        endpoints={[{ name: 'my-endpoint' } as any]}
        value={OPENAI_SELECTION}
        onChange={onChange}
      />,
    );

    await openDropdown();
    expect(screen.queryByTestId('model-option-endpoint-my-endpoint')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('model-group-gateway'));
    await userEvent.click(screen.getByTestId('model-option-endpoint-my-endpoint'));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ mode: 'endpoint', endpointName: 'my-endpoint' }));
  });

  test('providers are collapsed until expanded, then list their models in AI Gateway order', async () => {
    const onChange = jest.fn();
    renderWithDesignSystem(<IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={onChange} />);

    await openDropdown();
    expect(screen.queryByTestId('model-option-anthropic-claude-sonnet-4-6')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('model-provider-anthropic'));

    const modelOptions = screen.getAllByTestId(/^model-option-anthropic-/).map((el) => el.getAttribute('data-testid'));
    expect(modelOptions).toEqual([
      'model-option-anthropic-claude-opus-4-8',
      'model-option-anthropic-claude-sonnet-4-6',
      'model-option-anthropic-claude-sonnet-4-5',
    ]);

    await userEvent.click(screen.getByTestId('model-option-anthropic-claude-opus-4-8'));
    expect(onChange).toHaveBeenCalledWith({ mode: 'direct', provider: 'anthropic', model: 'claude-opus-4-8' });
  });

  test('falls back to the recommended model when the gateway returns none', async () => {
    renderWithDesignSystem(
      <IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={jest.fn()} />,
    );

    await openDropdown();
    await userEvent.click(screen.getByTestId('model-provider-openai'));
    expect(screen.getByTestId('model-option-openai-gpt-5.6-sol')).toBeInTheDocument();
  });

  test('all groups start collapsed when the dropdown opens', async () => {
    renderWithDesignSystem(
      <IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={jest.fn()} />,
    );

    await openDropdown();

    expect(screen.getByTestId('model-group-gateway')).toBeInTheDocument();
    expect(screen.getByTestId('model-provider-openai')).toBeInTheDocument();
    expect(screen.queryByTestId('model-option-openai-gpt-5.6-sol')).not.toBeInTheDocument();
    expect(screen.queryByTestId('model-create-endpoint')).not.toBeInTheDocument();
  });

  const CONNECTIONS_SECRET = {
    secret_id: 'sec-1',
    secret_name: 'my-openai',
    provider: 'openai',
    allowlisted_models: [{ provider: 'openai', model: 'gpt-5.6-sol' }],
  };

  test('shows five top-level groups: Existing connections, AI Gateway, then the three providers', async () => {
    jest.mocked(useSecretsQuery).mockReturnValue({
      data: [CONNECTIONS_SECRET],
      isLoading: false,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={jest.fn()} />,
    );
    await openDropdown();

    const groups = screen
      .getAllByTestId(/^(model-group-connections|model-group-gateway|model-provider-)/)
      .map((el) => el.getAttribute('data-testid'));
    expect(groups).toEqual([
      'model-group-connections',
      'model-group-gateway',
      'model-provider-openai',
      'model-provider-anthropic',
      'model-provider-gemini',
    ]);
    // Connection models are hidden until the group is expanded.
    expect(screen.queryByTestId('model-option-connection-sec-1-gpt-5.6-sol')).not.toBeInTheDocument();
  });

  test('omits Existing connections group when there are no allowlisted connection models', async () => {
    jest.mocked(useSecretsQuery).mockReturnValue({ data: [], isLoading: false, refetch: jest.fn() } as any);

    renderWithDesignSystem(
      <IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={jest.fn()} />,
    );
    await openDropdown();

    expect(screen.queryByTestId('model-group-connections')).not.toBeInTheDocument();
  });

  test('expands Existing connections to "model · name" and selecting one carries its secretId', async () => {
    const onChange = jest.fn();
    jest.mocked(useSecretsQuery).mockReturnValue({
      data: [CONNECTIONS_SECRET],
      isLoading: false,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(<IssueDetectionModelDropdown endpoints={[]} value={OPENAI_SELECTION} onChange={onChange} />);
    await openDropdown();

    await userEvent.click(screen.getByTestId('model-group-connections'));
    const option = screen.getByTestId('model-option-connection-sec-1-gpt-5.6-sol');
    expect(option).toHaveTextContent('gpt-5.6-sol · my-openai');

    await userEvent.click(option);
    expect(onChange).toHaveBeenCalledWith({
      mode: 'direct',
      provider: 'openai',
      model: 'gpt-5.6-sol',
      secretId: 'sec-1',
    });
  });
});
