import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useCreateEndpointForm } from './useCreateEndpointForm';
import { useCreateEndpointMutation } from './useCreateEndpointMutation';
import { useCreateSecret } from './useCreateSecret';
import { useCreateModelDefinitionMutation } from './useCreateModelDefinitionMutation';
import { useSetEndpointTagMutation } from './useSetEndpointTagMutation';
import { useModelsQuery } from './useModelsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useProviderConfigQuery } from './useProviderConfigQuery';
import { useSecretsQuery } from './useSecretsQuery';

jest.mock('./useCreateEndpointMutation');
jest.mock('./useCreateSecret');
jest.mock('./useCreateModelDefinitionMutation');
jest.mock('./useSetEndpointTagMutation');
jest.mock('./useModelsQuery');
jest.mock('./useEndpointsQuery');
jest.mock('./useProviderConfigQuery');
jest.mock('./useSecretsQuery');

const mockMutation = () => ({ mutateAsync: jest.fn(), error: null, isLoading: false, reset: jest.fn() }) as any;

describe('useCreateEndpointForm', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useCreateEndpointMutation).mockReturnValue(mockMutation());
    jest.mocked(useCreateSecret).mockReturnValue(mockMutation());
    jest.mocked(useCreateModelDefinitionMutation).mockReturnValue(mockMutation());
    jest.mocked(useSetEndpointTagMutation).mockReturnValue(mockMutation());
    jest.mocked(useModelsQuery).mockReturnValue({ data: [] } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [] } as any);
    jest.mocked(useSecretsQuery).mockReturnValue({ data: [] } as any);
    jest.mocked(useProviderConfigQuery).mockReturnValue({
      data: {
        default_mode: 'api_key',
        auth_modes: [{ mode: 'api_key', secret_fields: [{ name: 'api_key', required: true }], config_fields: [] }],
      },
    } as any);
  });

  test('isFormComplete is false when an API key contains non-ASCII characters', () => {
    const { result } = renderHook(() =>
      useCreateEndpointForm({ defaultProvider: 'openai', defaultModel: 'gpt-4o', defaultSecretName: 'my-key' }),
    );

    act(() => {
      result.current.form.setValue('newSecret.authMode', 'api_key');
      result.current.form.setValue('newSecret.secretFields', { api_key: '가나다' });
    });
    expect(result.current.isFormComplete).toBe(false);

    act(() => {
      result.current.form.setValue('newSecret.secretFields', { api_key: 'rkskek' });
    });
    expect(result.current.isFormComplete).toBe(true);
  });
});
