import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { CreateApiKeyModal } from './CreateApiKeyModal';
import { useCreateSecret } from '../../hooks/useCreateSecret';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import type { SecretFormData } from '../secrets';

jest.mock('../../hooks/useCreateSecret');
jest.mock('../../hooks/useProviderConfigQuery');
jest.mock('../create-endpoint', () => ({
  ProviderSelect: ({ onChange }: { onChange: (provider: string) => void }) => (
    <button onClick={() => onChange('openai')}>Select OpenAI</button>
  ),
}));
jest.mock('../secrets', () => ({
  SecretFormFields: ({ value, onChange }: { value: SecretFormData; onChange: (data: SecretFormData) => void }) => (
    <input
      aria-label="API key"
      value={value.secretFields['api_key'] ?? ''}
      onChange={(e) => onChange({ ...value, name: 'my-key', secretFields: { api_key: e.target.value } })}
    />
  ),
}));

describe('CreateApiKeyModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useCreateSecret).mockReturnValue({
      mutateAsync: jest.fn(),
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
    jest.mocked(useProviderConfigQuery).mockReturnValue({
      data: {
        default_mode: 'api_key',
        auth_modes: [{ mode: 'api_key', secret_fields: [{ name: 'api_key', required: true }], config_fields: [] }],
      },
    } as any);
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
});
