import { describe, expect, jest, beforeEach, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { CreateApiKeyModal } from './CreateApiKeyModal';
import { useCreateSecret } from '../../hooks/useCreateSecret';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import type { CreateSecretInfoResponse, CreateSecretRequest } from '../../types';

jest.mock('../../hooks/useCreateSecret');
jest.mock('../../hooks/useProviderConfigQuery');
jest.mock('../model-selector/ModelAllowlistField', () => ({
  ModelAllowlistField: () => <div data-testid="model-allowlist-field" />,
}));
jest.mock('../create-endpoint', () => ({
  ProviderSelect: ({ onChange }: { onChange: (provider: string) => void }) => (
    <button type="button" onClick={() => onChange('openai')}>
      Select OpenAI
    </button>
  ),
}));
jest.mock('../secrets', () => ({
  SecretFormFields: ({ value, onChange }: { value: any; onChange: (value: any) => void }) => (
    <div>
      <input aria-label="Key name" value={value.name} onChange={(e) => onChange({ ...value, name: e.target.value })} />
      <input
        aria-label="API key"
        value={value.secretFields.api_key ?? ''}
        onChange={(e) =>
          onChange({
            ...value,
            secretFields: { ...value.secretFields, api_key: e.target.value },
          })
        }
      />
    </div>
  ),
}));

describe('CreateApiKeyModal', () => {
  const createSecret = jest.fn<(request: CreateSecretRequest) => Promise<CreateSecretInfoResponse>>();

  beforeEach(() => {
    jest.clearAllMocks();
    createSecret.mockResolvedValue({
      secret: {
        secret_id: 's-1',
        secret_name: 'openai-key',
        masked_values: { api_key: 'sk-****test' },
        provider: 'openai',
        created_at: 1000,
        last_updated_at: 1000,
      },
    });
    jest.mocked(useCreateSecret).mockReturnValue({
      mutateAsync: createSecret,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
    jest.mocked(useProviderConfigQuery).mockReturnValue({
      data: {
        default_mode: 'api_key',
        auth_modes: [
          {
            mode: 'api_key',
            display_name: 'API key',
            secret_fields: [{ name: 'api_key', type: 'password', required: true }],
            config_fields: [],
          },
        ],
      },
    } as any);
  });

  test('allows creating a connection without allowlisted models', async () => {
    const onSuccess = jest.fn();
    renderWithDesignSystem(<CreateApiKeyModal open onClose={jest.fn()} onSuccess={onSuccess} />);

    await userEvent.click(screen.getByText('Select OpenAI'));
    await userEvent.type(screen.getByLabelText('Key name'), 'openai-key');
    await userEvent.type(screen.getByLabelText('API key'), 'sk-test');

    const submitButton = screen.getByRole('button', { name: 'Add connection' });
    expect(submitButton).not.toBeDisabled();

    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(createSecret).toHaveBeenCalledWith(
        expect.objectContaining({
          secret_name: 'openai-key',
          secret_value: { api_key: 'sk-test' },
          provider: 'openai',
          allowlisted_models: [],
        }),
      );
      expect(onSuccess).toHaveBeenCalled();
    });
  });
});
