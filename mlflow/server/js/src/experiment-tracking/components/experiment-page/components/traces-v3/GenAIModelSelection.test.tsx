import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { waitFor, fireEvent } from '@testing-library/react';
import React from 'react';
import { renderWithDesignSystem } from '../../../../../common/utils/TestUtils.react18';
import { GenAIModelSelection } from './GenAIModelSelection';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';

jest.mock('../../../../../gateway/hooks/useEndpointsQuery');
jest.mock('../../../../../gateway/hooks/useSecretsConfigQuery');
jest.mock('../../../../../gateway/components/create-endpoint/ModelSelect', () => ({
  ModelSelect: () => <div data-testid="model-select">Model Select</div>,
}));
jest.mock('./GenAIApiKeyConfigurator', () => ({
  GenAIApiKeyConfigurator: () => <div data-testid="api-key-configurator">API Key Configurator</div>,
}));
jest.mock('./GenAIAdvancedSettings', () => ({
  GenAIAdvancedSettings: () => <div data-testid="advanced-settings">Advanced Settings</div>,
}));
jest.mock('../../../../../gateway/components/endpoint-form', () => ({
  CreateEndpointModal: ({ open }: { open: boolean }) =>
    open ? <div data-testid="create-endpoint-modal">Create Endpoint Modal</div> : null,
}));
const mockUseApiKeyConfiguration = jest.fn();
jest.mock('../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration', () => ({
  useApiKeyConfiguration: (...args: any[]) => mockUseApiKeyConfiguration(...args),
}));

const mockEndpointQueryResult = (data: any[], isLoading = false) =>
  jest.mocked(useEndpointsQuery).mockReturnValue({
    data,
    isLoading,
    refetch: jest.fn(),
  } as any);

describe('GenAIModelSelection', () => {
  const defaultProps = {
    onValidityChange: jest.fn(),
    componentId: 'issue-detection-modal',
    description: 'Configure the model',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseApiKeyConfiguration.mockReturnValue({
      existingSecrets: [],
      authModes: [],
      defaultAuthMode: '',
      isLoadingProviderConfig: false,
    });
  });

  test('defaults to endpoint mode when endpoints exist', async () => {
    mockEndpointQueryResult([
      {
        endpoint_id: 'ep-1',
        name: 'test-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
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
      {
        endpoint_id: 'ep-1',
        name: 'test-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
    ]);

    const { getByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    // The first endpoint is auto-selected and shown in the trigger
    expect(getByText('test-endpoint')).toBeInTheDocument();
  });

  test('auto-selects first endpoint when endpoints are available', async () => {
    mockEndpointQueryResult([
      {
        endpoint_id: 'ep-1',
        name: 'first-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
      {
        endpoint_id: 'ep-2',
        name: 'second-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
    ]);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<GenAIModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      expect(ref.current?.getValues().endpointName).toBe('first-endpoint');
    });
  });

  test('auto-selects first existing API key when secrets are available', async () => {
    mockEndpointQueryResult([]);
    mockUseApiKeyConfiguration.mockReturnValue({
      existingSecrets: [
        { secret_id: 'secret-1', secret_name: 'My Key', provider: 'openai', masked_values: {} },
        { secret_id: 'secret-2', secret_name: 'Other Key', provider: 'openai', masked_values: {} },
      ],
      authModes: [],
      defaultAuthMode: '',
      isLoadingProviderConfig: false,
    });

    const ref = React.createRef<any>();
    renderWithDesignSystem(<GenAIModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      const config = ref.current?.getValues().apiKeyConfig;
      expect(config.mode).toBe('existing');
      expect(config.existingSecretId).toBe('secret-1');
    });
  });

  test('hides endpoint dropdown when no endpoints exist', () => {
    mockEndpointQueryResult([]);

    const { queryByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    expect(queryByText('Select endpoint')).not.toBeInTheDocument();
  });

  test('respects initialValues.endpointName without overriding it when mode is not set', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [
        { endpoint_id: 'ep-1', name: 'first-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
        { endpoint_id: 'ep-2', name: 'specific-endpoint', model_mappings: [], created_at: 0, last_updated_at: 0 },
      ],
      isLoading: false,
    } as any);

    const ref = React.createRef<any>();
    renderWithDesignSystem(
      <GenAIModelSelection {...defaultProps} ref={ref} initialValues={{ endpointName: 'specific-endpoint' }} />,
    );

    await waitFor(() => {
      const values = ref.current?.getValues();
      // Should not be overridden with first-endpoint, and mode should be inferred as 'endpoint'
      expect(values.endpointName).toBe('specific-endpoint');
      expect(values.mode).toBe('endpoint');
    });
  });

  test('initializes saveKey from initialValues.saveKey', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [], isLoading: false } as any);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<GenAIModelSelection {...defaultProps} ref={ref} initialValues={{ saveKey: false }} />);

    await waitFor(() => {
      expect(ref.current?.getValues().saveKey).toBe(false);
    });
  });

  test('does not auto-switch apiKeyConfig when readOnly is true', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [], isLoading: false } as any);
    mockUseApiKeyConfiguration.mockReturnValue({
      existingSecrets: [{ secret_id: 'existing-secret', secret_name: 'My Key', provider: 'openai', masked_values: {} }],
      authModes: [],
      defaultAuthMode: '',
      isLoadingProviderConfig: false,
    });

    const ref = React.createRef<any>();
    renderWithDesignSystem(<GenAIModelSelection {...defaultProps} ref={ref} readOnly />);

    // Wait for effects to settle
    await waitFor(() => {
      const config = ref.current?.getValues().apiKeyConfig;
      // Should remain 'new' because readOnly skips the auto-switch
      expect(config.mode).toBe('new');
    });
  });

  test('does not auto-switch apiKeyConfig when initialValues.apiKeyConfig is provided', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [], isLoading: false } as any);
    mockUseApiKeyConfiguration.mockReturnValue({
      existingSecrets: [{ secret_id: 'existing-secret', secret_name: 'My Key', provider: 'openai', masked_values: {} }],
      authModes: [],
      defaultAuthMode: '',
      isLoadingProviderConfig: false,
    });

    const initialApiKeyConfig = {
      mode: 'new' as const,
      existingSecretId: '',
      newSecret: { name: 'provided-key', authMode: '', secretFields: {}, configFields: {} },
    };

    const ref = React.createRef<any>();
    renderWithDesignSystem(
      <GenAIModelSelection {...defaultProps} ref={ref} initialValues={{ apiKeyConfig: initialApiKeyConfig }} />,
    );

    await waitFor(() => {
      const config = ref.current?.getValues().apiKeyConfig;
      // Should not be auto-switched to 'existing' because initialValues.apiKeyConfig was provided
      expect(config.mode).toBe('new');
      expect(config.newSecret.name).toBe('provided-key');
    });
  });

  test('shows "Configure model directly" option when showConfigureDirectly is true', () => {
    mockEndpointQueryResult([
      {
        endpoint_id: 'ep-1',
        name: 'test-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
    ]);

    const { getByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} showConfigureDirectly />);

    fireEvent.click(getByText('test-endpoint'));
    expect(getByText('Configure model directly')).toBeInTheDocument();
  });

  test('hides "Configure model directly" option by default', () => {
    mockEndpointQueryResult([
      {
        endpoint_id: 'ep-1',
        name: 'test-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
    ]);

    const { getByText, queryByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    fireEvent.click(getByText('test-endpoint'));
    expect(queryByText('Configure model directly')).not.toBeInTheDocument();
  });

  test('shows "Create Gateway endpoint" option when showCreateEndpoint is true', () => {
    mockEndpointQueryResult([
      {
        endpoint_id: 'ep-1',
        name: 'test-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
    ]);

    const { getByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} showCreateEndpoint />);

    fireEvent.click(getByText('test-endpoint'));
    expect(getByText('Create Gateway endpoint')).toBeInTheDocument();
  });

  test('hides "Create Gateway endpoint" option by default', () => {
    mockEndpointQueryResult([
      {
        endpoint_id: 'ep-1',
        name: 'test-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
    ]);

    const { getByText, queryByText } = renderWithDesignSystem(<GenAIModelSelection {...defaultProps} />);

    fireEvent.click(getByText('test-endpoint'));
    expect(queryByText('Create Gateway endpoint')).not.toBeInTheDocument();
  });

  test('opens create endpoint modal when "Create Gateway endpoint" is clicked', () => {
    mockEndpointQueryResult([
      {
        endpoint_id: 'ep-1',
        name: 'test-endpoint',
        model_mappings: [],
        created_at: 0,
        last_updated_at: 0,
      },
    ]);

    const { getByText, queryByTestId } = renderWithDesignSystem(
      <GenAIModelSelection {...defaultProps} showCreateEndpoint />,
    );

    fireEvent.click(getByText('test-endpoint'));
    expect(queryByTestId('create-endpoint-modal')).not.toBeInTheDocument();
    fireEvent.click(getByText('Create Gateway endpoint'));
    expect(queryByTestId('create-endpoint-modal')).toBeInTheDocument();
  });
});
