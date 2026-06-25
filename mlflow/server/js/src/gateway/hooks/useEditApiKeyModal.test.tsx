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
