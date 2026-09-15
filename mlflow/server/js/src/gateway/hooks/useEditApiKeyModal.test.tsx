import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import React from 'react';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useEditApiKeyModal } from './useEditApiKeyModal';
import { useUpdateSecret } from './useUpdateSecret';
import { useProviderConfigQuery } from './useProviderConfigQuery';
import type { SecretInfo, UpdateSecretRequest } from '../types';

jest.mock('./useUpdateSecret');
jest.mock('./useProviderConfigQuery');

const mockUpdateSecret = jest.fn<(request: UpdateSecretRequest) => Promise<void>>();
const mockResetMutation = jest.fn();

const mockSecret: SecretInfo = {
  secret_id: 's-1',
  secret_name: 'openai-key',
  provider: 'openai',
  masked_values: { api_key: 'sk-****1234' },
  created_at: 1000,
  last_updated_at: 1000,
  auth_config: { auth_mode: 'api_key', base_url: 'https://api.openai.com' },
};

const mockApiBaseSecret: SecretInfo = {
  ...mockSecret,
  auth_config: { auth_mode: 'api_key', api_base: 'https://api.openai.com/v1' },
};

const mockPortkeySecret: SecretInfo = {
  ...mockApiBaseSecret,
  secret_name: 'portkey-key',
  provider: 'portkey',
  masked_values: {
    api_key: 'pk-****1234',
    portkey_config: 'pc-****5678',
    provider_api_key: 'sk-****9012',
  },
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <IntlProvider locale="en">{children}</IntlProvider>
    </QueryClientProvider>
  );
}

describe('useEditApiKeyModal', () => {
  const mockOnClose = jest.fn();
  const mockOnSuccess = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUpdateSecret.mockResolvedValue(undefined);
    jest.mocked(useUpdateSecret).mockReturnValue({
      mutateAsync: mockUpdateSecret,
      isLoading: false,
      error: null,
      reset: mockResetMutation,
    } as any);
    jest.mocked(useProviderConfigQuery).mockReturnValue({ data: undefined } as any);
  });

  test('initializes form from secret data', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    expect(result.current.formData.name).toBe('openai-key');
    expect(result.current.formData.authMode).toBe('api_key');
    expect(result.current.formData.configFields).toEqual({ base_url: 'https://api.openai.com' });
    expect(result.current.formData.secretFields).toEqual({});
  });

  test('isDirty is false initially', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    expect(result.current.isDirty).toBe(false);
  });

  test('isDirty becomes true when config fields change', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { base_url: 'https://new-url.com' },
      });
    });

    expect(result.current.isDirty).toBe(true);
  });

  test('isDirty becomes true when secret fields change', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        secretFields: { api_key: 'new-key-value' },
      });
    });

    expect(result.current.isDirty).toBe(true);
  });

  test('isFormValid is false when not dirty', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    expect(result.current.isFormValid).toBe(false);
  });

  test('isFormValid is true when dirty with existing secret', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { base_url: 'https://new-url.com' },
      });
    });

    expect(result.current.isFormValid).toBe(true);
  });

  test('isFormValid is false when api_base changes without secret values', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockApiBaseSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { api_base: 'https://proxy.example.com/v1' },
      });
    });

    expect(result.current.isFormValid).toBe(false);
  });

  test('resetForm reverts to initial form data', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { base_url: 'https://new-url.com' },
      });
    });
    expect(result.current.isDirty).toBe(true);

    act(() => {
      result.current.resetForm();
    });

    expect(result.current.isDirty).toBe(false);
    expect(result.current.formData.configFields).toEqual({ base_url: 'https://api.openai.com' });
  });

  test('handleSubmit sends only config when no secret values entered', async () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { base_url: 'https://new-url.com' },
      });
    });

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(mockUpdateSecret).toHaveBeenCalledWith(
      expect.objectContaining({
        secret_id: 's-1',
        secret_value: undefined,
        auth_config: expect.objectContaining({ base_url: 'https://new-url.com' }),
      }),
    );
    expect(mockOnSuccess).toHaveBeenCalled();
  });

  test('handleSubmit rejects api_base change when no secret values entered', async () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockApiBaseSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { api_base: 'https://proxy.example.com/v1' },
      });
    });

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(mockUpdateSecret).not.toHaveBeenCalled();
    expect(result.current.errors.secretFields?.['api_key']).toBe(
      'Re-enter this credential when changing the API Base URL.',
    );
  });

  test('handleSubmit includes secret_value when user enters values', async () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        secretFields: { api_key: 'sk-new-key' },
      });
    });

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(mockUpdateSecret).toHaveBeenCalledWith(
      expect.objectContaining({
        secret_id: 's-1',
        secret_value: { api_key: 'sk-new-key' },
      }),
    );
  });

  test('handleSubmit includes secret_value when api_base changes with secret values', async () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockApiBaseSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        secretFields: { api_key: 'sk-new-key' },
        configFields: { api_base: 'https://proxy.example.com/v1' },
      });
    });

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(mockUpdateSecret).toHaveBeenCalledWith(
      expect.objectContaining({
        secret_id: 's-1',
        secret_value: { api_key: 'sk-new-key' },
        auth_config: expect.objectContaining({ api_base: 'https://proxy.example.com/v1' }),
      }),
    );
  });

  test('api_base change requires all existing optional secret values', async () => {
    jest.mocked(useProviderConfigQuery).mockReturnValue({
      data: {
        default_mode: 'api_key',
        auth_modes: [
          {
            mode: 'api_key',
            display_name: 'API Key',
            secret_fields: [
              { name: 'api_key', type: 'string', required: true },
              { name: 'portkey_config', type: 'string', required: false },
              { name: 'provider_api_key', type: 'string', required: false },
            ],
            config_fields: [{ name: 'api_base', type: 'string', required: false }],
          },
        ],
      },
    } as any);
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockPortkeySecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        secretFields: { api_key: 'pk-new-key' },
        configFields: { api_base: 'https://proxy.example.com/v1' },
      });
    });

    expect(result.current.isFormValid).toBe(false);

    await act(async () => {
      await result.current.handleSubmit();
    });
    expect(mockUpdateSecret).not.toHaveBeenCalled();
    expect(result.current.errors.secretFields?.['portkey_config']).toBe(
      'Re-enter this credential when changing the API Base URL.',
    );
    expect(result.current.errors.secretFields?.['provider_api_key']).toBe(
      'Re-enter this credential when changing the API Base URL.',
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        secretFields: {
          api_key: 'pk-new-key',
          portkey_config: 'pc-new-config',
          provider_api_key: 'sk-new-provider-key',
        },
      });
    });

    expect(result.current.isFormValid).toBe(true);

    await act(async () => {
      await result.current.handleSubmit();
    });
    expect(mockUpdateSecret).toHaveBeenCalledWith(
      expect.objectContaining({
        secret_value: {
          api_key: 'pk-new-key',
          portkey_config: 'pc-new-config',
          provider_api_key: 'sk-new-provider-key',
        },
      }),
    );
  });

  test('handleSubmit resets isDirty after successful save', async () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { base_url: 'https://new-url.com' },
      });
    });
    expect(result.current.isDirty).toBe(true);

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(result.current.isDirty).toBe(false);
  });

  test('handleSubmit does not call onSuccess on API error', async () => {
    mockUpdateSecret.mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleFormDataChange({
        ...result.current.formData,
        configFields: { base_url: 'https://new-url.com' },
      });
    });

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(mockOnSuccess).not.toHaveBeenCalled();
  });

  test('handleClose resets form to empty and calls onClose', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    act(() => {
      result.current.handleClose();
    });

    expect(result.current.formData.name).toBe('');
    expect(mockOnClose).toHaveBeenCalled();
  });

  test('provider is derived from secret', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: mockSecret, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    expect(result.current.provider).toBe('openai');
  });

  test('provider is empty string when no secret', () => {
    const { result } = renderHook(
      () => useEditApiKeyModal({ secret: null, onClose: mockOnClose, onSuccess: mockOnSuccess }),
      { wrapper: createWrapper() },
    );

    expect(result.current.provider).toBe('');
  });
});
