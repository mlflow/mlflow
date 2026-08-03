import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModelDropdown, type IssueDetectionModelSelection } from './IssueDetectionModelDropdown';
import { useModelsQuery } from '../../../../../gateway/hooks/useModelsQuery';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';

jest.mock('../../../../../gateway/hooks/useModelsQuery', () => ({
  useModelsQuery: jest.fn(),
}));
jest.mock('../../../../../gateway/hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: jest.fn(),
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
});
