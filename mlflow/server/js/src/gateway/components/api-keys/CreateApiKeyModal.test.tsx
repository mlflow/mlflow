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
    <>
      <button onClick={() => onChange('openai')}>Select OpenAI</button>
      <button onClick={() => onChange('vertex_ai')}>Select Vertex AI</button>
    </>
  ),
}));
let mockAuthMode = 'default_chain';
jest.mock('../secrets', () => ({
  SecretFormFields: ({ value, onChange }: { value: SecretFormData; onChange: (data: SecretFormData) => void }) => (
    <>
      <input
        aria-label="API key"
        value={value.secretFields['api_key'] ?? ''}
        onChange={(e) => onChange({ ...value, name: 'my-key', secretFields: { api_key: e.target.value } })}
      />
      <button
        onClick={() =>
          onChange({
            ...value,
            name: 'vertex-adc',
            authMode: mockAuthMode,
            configFields: { vertex_project: 'my-project', vertex_location: 'us-central1' },
          })
        }
      >
        Fill form
      </button>
    </>
  ),
}));

const providerConfigs = {
  openai: {
    default_mode: 'api_key',
    auth_modes: [{ mode: 'api_key', secret_fields: [{ name: 'api_key', required: true }], config_fields: [] }],
  },
  vertex_ai: {
    default_mode: 'service_account_json',
    auth_modes: [
      {
        mode: 'service_account_json',
        display_name: 'Service Account JSON',
        // Optional so the submit button stays enabled and validateForm decides
        secret_fields: [{ name: 'vertex_credentials', type: 'string', required: false }],
        config_fields: [{ name: 'vertex_project', type: 'string', required: true }],
      },
      {
        mode: 'default_chain',
        display_name: 'Application Default Credentials',
        secret_fields: [],
        config_fields: [{ name: 'vertex_project', type: 'string', required: true }],
      },
    ],
  },
};

describe('CreateApiKeyModal', () => {
  const mockCreateSecret = jest.fn<(request: any) => Promise<void>>();

  beforeEach(() => {
    jest.clearAllMocks();
    mockAuthMode = 'default_chain';
    mockCreateSecret.mockResolvedValue(undefined);
    jest.mocked(useCreateSecret).mockReturnValue({
      mutateAsync: mockCreateSecret,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
    jest
      .mocked(useProviderConfigQuery)
      .mockImplementation(
        ({ provider }) => ({ data: providerConfigs[provider as keyof typeof providerConfigs] }) as any,
      );
  });

  test('disables creating a key that contains non-ASCII characters', async () => {
    renderWithDesignSystem(<CreateApiKeyModal open onClose={jest.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Select OpenAI' }));

    const input = screen.getByLabelText('API key');
    await userEvent.type(input, '가나다');
    expect(screen.getByRole('button', { name: 'Create API Key' })).toBeDisabled();

    await userEvent.clear(input);
    await userEvent.type(input, 'rkskek');
    expect(screen.getByRole('button', { name: 'Create API Key' })).toBeEnabled();
  });

  test('creates a key for an auth mode without secret fields', async () => {
    const onSuccess = jest.fn();
    renderWithDesignSystem(<CreateApiKeyModal open onClose={jest.fn()} onSuccess={onSuccess} />);

    await userEvent.click(screen.getByText('Select Vertex AI'));
    await userEvent.click(screen.getByText('Fill form'));
    await userEvent.click(screen.getByRole('button', { name: 'Create API Key' }));

    await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    expect(mockCreateSecret).toHaveBeenCalledWith({
      secret_name: 'vertex-adc',
      secret_value: {},
      provider: 'vertex_ai',
      auth_config: { auth_mode: 'default_chain', vertex_project: 'my-project', vertex_location: 'us-central1' },
    });
  });

  test('does not create a key without secret values for an auth mode with secret fields', async () => {
    mockAuthMode = 'service_account_json';
    const onSuccess = jest.fn();
    renderWithDesignSystem(<CreateApiKeyModal open onClose={jest.fn()} onSuccess={onSuccess} />);

    await userEvent.click(screen.getByText('Select Vertex AI'));
    await userEvent.click(screen.getByText('Fill form'));
    await userEvent.click(screen.getByRole('button', { name: 'Create API Key' }));

    expect(mockCreateSecret).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
  });
});
