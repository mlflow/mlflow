import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { CreateApiKeyModal } from './CreateApiKeyModal';
import { useCreateSecret } from '../../hooks/useCreateSecret';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import type { SecretFormData } from '../secrets';

jest.mock('../../hooks/useCreateSecret');
jest.mock('../../hooks/useProviderConfigQuery');
jest.mock('../create-endpoint', () => ({
  ProviderSelect: ({ onChange }: { onChange: (provider: string) => void }) => (
    <button onClick={() => onChange('vertex_ai')}>Select Vertex AI</button>
  ),
}));
jest.mock('../secrets', () => ({
  SecretFormFields: ({ value, onChange }: { value: SecretFormData; onChange: (data: SecretFormData) => void }) => (
    <button
      onClick={() =>
        onChange({
          ...value,
          name: 'vertex-adc',
          authMode: 'default_chain',
          configFields: { vertex_project: 'my-project', vertex_location: 'us-central1' },
        })
      }
    >
      Fill ADC form
    </button>
  ),
}));

const vertexProviderConfig = {
  default_mode: 'service_account_json',
  auth_modes: [
    {
      mode: 'service_account_json',
      display_name: 'Service Account JSON',
      secret_fields: [{ name: 'vertex_credentials', type: 'string', required: true }],
      config_fields: [{ name: 'vertex_project', type: 'string', required: true }],
    },
    {
      mode: 'default_chain',
      display_name: 'Application Default Credentials',
      secret_fields: [],
      config_fields: [{ name: 'vertex_project', type: 'string', required: true }],
    },
  ],
};

describe('CreateApiKeyModal', () => {
  const mockCreateSecret = jest.fn<(request: any) => Promise<void>>();

  beforeEach(() => {
    jest.clearAllMocks();
    mockCreateSecret.mockResolvedValue(undefined);
    jest.mocked(useCreateSecret).mockReturnValue({
      mutateAsync: mockCreateSecret,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
    jest.mocked(useProviderConfigQuery).mockReturnValue({ data: vertexProviderConfig } as any);
  });

  test('creates a key for an auth mode without secret fields', async () => {
    const onSuccess = jest.fn();
    renderWithDesignSystem(<CreateApiKeyModal open onClose={jest.fn()} onSuccess={onSuccess} />);

    await userEvent.click(screen.getByText('Select Vertex AI'));
    await userEvent.click(screen.getByText('Fill ADC form'));
    await userEvent.click(screen.getByRole('button', { name: 'Create API Key' }));

    await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    expect(mockCreateSecret).toHaveBeenCalledWith({
      secret_name: 'vertex-adc',
      secret_value: {},
      provider: 'vertex_ai',
      auth_config: { auth_mode: 'default_chain', vertex_project: 'my-project', vertex_location: 'us-central1' },
    });
  });
});
