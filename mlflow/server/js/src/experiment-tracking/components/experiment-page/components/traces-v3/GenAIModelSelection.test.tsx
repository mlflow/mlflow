import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { waitFor, fireEvent } from '@testing-library/react';
import React from 'react';
import { renderWithDesignSystem } from '../../../../../common/utils/TestUtils.react18';
import { GenAIModelSelection } from './GenAIModelSelection';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { useSecretsQuery } from '../../../../../gateway/hooks/useSecretsQuery';

jest.mock('../../../../../gateway/hooks/useEndpointsQuery');
jest.mock('../../../../../gateway/hooks/useSecretsQuery');
jest.mock('../../../../../gateway/hooks/useSecretsConfigQuery');
jest.mock('../../../../../gateway/components/endpoint-form', () => ({
  CreateEndpointModal: ({ open }: { open: boolean }) =>
    open ? <div data-testid="create-endpoint-modal">Create Endpoint Modal</div> : null,
}));

const mockNavigate = jest.fn();
jest.mock('../../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/RoutingUtils')>(
    '../../../../../common/utils/RoutingUtils',
  ),
  useNavigate: () => mockNavigate,
}));

const mockEndpointQueryResult = (data: any[], isLoading = false) =>
  jest.mocked(useEndpointsQuery).mockReturnValue({
    data,
    isLoading,
    refetch: jest.fn(),
  } as any);

const mockSecretsQueryResult = (secrets: any[], isLoading = false) =>
  jest.mocked(useSecretsQuery).mockReturnValue({
    data: secrets,
    isLoading,
    error: undefined,
    refetch: jest.fn(),
  } as any);

// A secret carrying two allowlisted models -> two selectable pairs.
const SECRET_WITH_MODELS = {
  secret_id: 'secret-1',
  secret_name: 'My Anthropic Key',
  provider: 'anthropic',
  masked_values: {},
  created_at: 0,
  last_updated_at: 0,
  allowlisted_models: [
    { model: 'claude-sonnet-4-6', provider: 'anthropic', supports_function_calling: true },
    { model: 'claude-opus-4', provider: 'anthropic', supports_function_calling: true },
  ],
};

describe('GenAIModelSelection', () => {
  const defaultProps = {
    onValidityChange: jest.fn(),
    componentId: 'issue-detection-modal',
    description: 'Configure the model',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockSecretsQueryResult([]);
  });

  test('defaults to endpoint mode when endpoints exist', async () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'test-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<GenAIModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      expect(ref.current?.getValues().mode).toBe('endpoint');
    });
  });

  test('defaults to direct mode when no endpoints exist', async () => {
    mockEndpointQueryResult([]);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<GenAIModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      expect(ref.current?.getValues().mode).toBe('direct');
    });
  });

  test('shows loading state while fetching endpoints', async () => {
    mockEndpointQueryResult([], true);

    const { getByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    expect(getByText('Loading endpoints...')).toBeInTheDocument();
  });

  test('shows endpoint dropdown when endpoints exist', () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'test-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const { getByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    expect(getByText('test-endpoint')).toBeInTheDocument();
  });

  test('auto-selects first endpoint when endpoints are available', async () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'first-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
      { endpoint_id: 'ep-2', name: 'second-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<GenAIModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      expect(ref.current?.getValues().endpointName).toBe('first-endpoint');
    });
  });

  test('hides endpoint dropdown when no endpoints exist', () => {
    mockEndpointQueryResult([]);

    const { queryByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    expect(queryByText('Select endpoint')).not.toBeInTheDocument();
  });

  test('respects initialValues.endpointName without overriding it when mode is not set', async () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'first-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
      { endpoint_id: 'ep-2', name: 'specific-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const ref = React.createRef<any>();
    renderWithDesignSystem(
      <GenAIModelSelection {...defaultProps} ref={ref} initialValues={{ endpointName: 'specific-endpoint' }} />,
    );

    await waitFor(() => {
      const values = ref.current?.getValues();
      expect(values.endpointName).toBe('specific-endpoint');
      expect(values.mode).toBe('endpoint');
    });
  });

  test('shows "Configure model directly" option when showConfigureDirectly is true', () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'test-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const { getByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} showConfigureDirectly />);

    fireEvent.click(getByText('test-endpoint'));
    expect(getByText('Configure model directly')).toBeInTheDocument();
  });

  test('hides "Configure model directly" option by default', () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'test-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const { getByText, queryByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    fireEvent.click(getByText('test-endpoint'));
    expect(queryByText('Configure model directly')).not.toBeInTheDocument();
  });

  test('shows "Create Gateway endpoint" option when showCreateEndpoint is true', () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'test-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const { getByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} showCreateEndpoint />);

    fireEvent.click(getByText('test-endpoint'));
    expect(getByText('Create Gateway endpoint')).toBeInTheDocument();
  });

  test('opens create endpoint modal when "Create Gateway endpoint" is clicked', () => {
    mockEndpointQueryResult([
      { endpoint_id: 'ep-1', name: 'test-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
    ]);

    const { getByText, queryByTestId } = renderWithDesignSystem(
      <GenAIModelSelection {...defaultProps} showCreateEndpoint />,
    );

    fireEvent.click(getByText('test-endpoint'));
    expect(queryByTestId('create-endpoint-modal')).not.toBeInTheDocument();
    fireEvent.click(getByText('Create Gateway endpoint'));
    expect(queryByTestId('create-endpoint-modal')).toBeInTheDocument();
  });

  describe('direct mode: allowlisted-pair dropdown', () => {
    beforeEach(() => {
      // No endpoints -> defaults to direct mode.
      mockEndpointQueryResult([]);
    });

    test('auto-selects the first allowlisted pair and exposes an existing-secret contract', async () => {
      mockSecretsQueryResult([SECRET_WITH_MODELS]);

      const ref = React.createRef<any>();
      renderWithDesignSystem(<GenAIModelSelection {...defaultProps} showConfigureDirectly ref={ref} />);

      await waitFor(() => {
        const values = ref.current?.getValues();
        expect(values.mode).toBe('direct');
        // Pairs are sorted by label; "Anthropic · claude-opus-4" sorts before "... claude-sonnet-4-6".
        expect(values.provider).toBe('anthropic');
        expect(values.model).toBe('claude-opus-4');
        expect(values.saveKey).toBe(false);
        expect(values.apiKeyConfig.mode).toBe('existing');
        expect(values.apiKeyConfig.existingSecretId).toBe('secret-1');
      });
    });

    test('is valid once a pair is auto-selected', async () => {
      mockSecretsQueryResult([SECRET_WITH_MODELS]);

      const ref = React.createRef<any>();
      renderWithDesignSystem(<GenAIModelSelection {...defaultProps} showConfigureDirectly ref={ref} />);

      await waitFor(() => {
        expect(ref.current?.isValid).toBe(true);
      });
    });

    test('renders each allowlisted pair as an option', async () => {
      mockSecretsQueryResult([SECRET_WITH_MODELS]);

      const { getByText, getAllByText } = renderWithDesignSystem(
        <GenAIModelSelection {...defaultProps} showConfigureDirectly />,
      );

      // The auto-selected pair is shown in the trigger; open the dropdown to see all options.
      fireEvent.click(getAllByText('Anthropic · claude-opus-4')[0]);
      expect(getByText('Anthropic · claude-sonnet-4-6')).toBeInTheDocument();
    });

    test('shows an empty state with an "Add a connection" action when no pairs exist', async () => {
      mockSecretsQueryResult([]);

      const ref = React.createRef<any>();
      const { getByText } = renderWithDesignSystem(
        <GenAIModelSelection {...defaultProps} showConfigureDirectly ref={ref} />,
      );

      fireEvent.click(getByText('Select a model'));
      expect(
        getByText('No models available. Add a connection with allowed models to get started.'),
      ).toBeInTheDocument();
      fireEvent.click(getByText('Add a connection'));
      expect(mockNavigate).toHaveBeenCalledWith(expect.stringContaining('#llm-connections'));

      // With no pairs, the direct mode is invalid.
      await waitFor(() => {
        expect(ref.current?.isValid).toBe(false);
      });
    });

    test('"Manage connections" footer link navigates to the connections settings', async () => {
      mockSecretsQueryResult([SECRET_WITH_MODELS]);

      const { getAllByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} showConfigureDirectly />);

      // Footer link is rendered below the dropdown.
      fireEvent.click(getAllByText('Manage connections')[0]);
      expect(mockNavigate).toHaveBeenCalledWith(expect.stringContaining('#llm-connections'));
    });
  });
});
