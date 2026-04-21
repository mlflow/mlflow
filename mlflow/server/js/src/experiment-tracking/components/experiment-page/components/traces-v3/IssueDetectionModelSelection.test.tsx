import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { waitFor } from '@testing-library/react';
import React from 'react';
import { renderWithDesignSystem } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModelSelection } from './IssueDetectionModelSelection';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';

jest.mock('../../../../../gateway/hooks/useEndpointsQuery');
jest.mock('../../../../../gateway/hooks/useSecretsConfigQuery');
jest.mock('../../../../../gateway/components/create-endpoint/ModelSelect', () => ({
  ModelSelect: () => <div data-testid="model-select">Model Select</div>,
}));
jest.mock('./IssueDetectionApiKeyConfigurator', () => ({
  IssueDetectionApiKeyConfigurator: () => <div data-testid="api-key-configurator">API Key Configurator</div>,
}));
jest.mock('./IssueDetectionAdvancedSettings', () => ({
  IssueDetectionAdvancedSettings: () => <div data-testid="advanced-settings">Advanced Settings</div>,
}));
const mockUseApiKeyConfiguration = jest.fn();
jest.mock('../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration', () => ({
  useApiKeyConfiguration: (...args: any[]) => mockUseApiKeyConfiguration(...args),
}));

describe('IssueDetectionModelSelection', () => {
  const defaultProps = {
    selectedTraceIds: [],
    onSelectTracesClick: jest.fn(),
    onValidityChange: jest.fn(),
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
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [
        {
          endpoint_id: 'ep-1',
          name: 'test-endpoint',
          model_mappings: [],
          created_at: 0,
          last_updated_at: 0,
        },
      ],
      isLoading: false,
    } as any);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<IssueDetectionModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      expect(ref.current?.getValues().mode).toBe('endpoint');
    });
  });

  test('defaults to direct mode when no endpoints exist', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<IssueDetectionModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      expect(ref.current?.getValues().mode).toBe('direct');
    });
  });

  test('shows loading state while fetching endpoints', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: true,
    } as any);

    const { getByText } = renderWithDesignSystem(<IssueDetectionModelSelection {...defaultProps} />);

    expect(getByText('Loading endpoints...')).toBeInTheDocument();
  });

  test('shows endpoint dropdown when endpoints exist', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [
        {
          endpoint_id: 'ep-1',
          name: 'test-endpoint',
          model_mappings: [],
          created_at: 0,
          last_updated_at: 0,
        },
      ],
      isLoading: false,
    } as any);

    const { getByText } = renderWithDesignSystem(<IssueDetectionModelSelection {...defaultProps} />);

    // The first endpoint is auto-selected and shown in the trigger
    expect(getByText('test-endpoint')).toBeInTheDocument();
  });

  test('auto-selects first endpoint when endpoints are available', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [
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
      ],
      isLoading: false,
    } as any);

    const ref = React.createRef<any>();
    renderWithDesignSystem(<IssueDetectionModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      expect(ref.current?.getValues().endpointName).toBe('first-endpoint');
    });
  });

  test('auto-selects first existing API key when secrets are available', async () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);
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
    renderWithDesignSystem(<IssueDetectionModelSelection {...defaultProps} ref={ref} />);

    await waitFor(() => {
      const config = ref.current?.getValues().apiKeyConfig;
      expect(config.mode).toBe('existing');
      expect(config.existingSecretId).toBe('secret-1');
    });
  });

  test('hides endpoint dropdown when no endpoints exist', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    const { queryByText } = renderWithDesignSystem(<IssueDetectionModelSelection {...defaultProps} />);

    expect(queryByText('Select endpoint')).not.toBeInTheDocument();
  });
});
