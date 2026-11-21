import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useCreateEndpointForm, useCreateEndpointFormComplete } from './useCreateEndpointForm';
import { useCreateEndpointMutation } from './useCreateEndpointMutation';
import { useCreateSecret } from './useCreateSecret';
import { useCreateModelDefinitionMutation } from './useCreateModelDefinitionMutation';
import { useSetEndpointTagMutation } from './useSetEndpointTagMutation';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useProviderConfigQuery } from './useProviderConfigQuery';
import { useSecretsQuery } from './useSecretsQuery';

jest.mock('./useCreateEndpointMutation');
jest.mock('./useCreateSecret');
jest.mock('./useCreateModelDefinitionMutation');
jest.mock('./useSetEndpointTagMutation');
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
    const { result } = renderHook(() => {
      const state = useCreateEndpointForm({
        defaultProvider: 'openai',
        defaultModel: 'gpt-4o',
        defaultSecretName: 'my-key',
      });
      return { ...state, isFormComplete: useCreateEndpointFormComplete(state.form.control) };
    });

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

  // Editing credentials should only update sections that consume those values.
  test('does not rerender the form shell for name or credential edits', () => {
    const renderShell = jest.fn();
    const { result } = renderHook(() => {
      renderShell();
      return useCreateEndpointForm({ defaultProvider: 'openai', defaultModel: 'gpt-4o' });
    });
    renderShell.mockClear();
    act(() => result.current.form.setValue('name', 'my-endpoint'));
    act(() => result.current.form.setValue('newSecret.name', 'my-key'));
    act(() => result.current.form.setValue('newSecret.secretFields', { api_key: 'synthetic-key' }));
    expect(renderShell).not.toHaveBeenCalled();
  });
  test('does not rerender completeness for name edits on ordinary endpoints', () => {
    const renderCompleteness = jest.fn();
    const { result } = renderHook(() => {
      const state = useCreateEndpointForm({ defaultProvider: 'openai', defaultModel: 'model-1' });
      renderCompleteness();
      return { ...state, complete: useCreateEndpointFormComplete(state.form.control) };
    });
    renderCompleteness.mockClear();
    act(() => result.current.form.setValue('name', 'my-endpoint'));
    expect(renderCompleteness).not.toHaveBeenCalled();
  });

  test('refreshes completeness when switching between new and existing credentials', () => {
    const { result } = renderHook(() => {
      const state = useCreateEndpointForm({ defaultProvider: 'openai', defaultModel: 'model-1' });
      return { ...state, complete: useCreateEndpointFormComplete(state.form.control) };
    });
    expect(result.current.complete).toBe(false);
    act(() => result.current.form.setValue('secretMode', 'existing'));
    expect(result.current.complete).toBe(false);
    act(() => result.current.form.setValue('existingSecretId', 'secret-1'));
    expect(result.current.complete).toBe(true);
    act(() => result.current.form.setValue('secretMode', 'new'));
    expect(result.current.complete).toBe(false);
    act(() =>
      result.current.form.setValue('newSecret', {
        name: 'key',
        authMode: 'api_key',
        secretFields: { api_key: 'synthetic' },
        configFields: {},
      }),
    );
    expect(result.current.complete).toBe(true);
    act(() => result.current.form.setValue('provider', 'anthropic'));
    expect(result.current.complete).toBe(false);
    expect(result.current.form.getValues()).toMatchObject({
      modelName: '',
      secretMode: 'new',
      existingSecretId: '',
      newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
    });
  });

  test('keeps coding-agent name requirements and duplicate-name validation', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [{ name: 'taken' }] } as any);
    const { result } = renderHook(() => {
      const state = useCreateEndpointForm({ codingAgent: 'codex' });
      return { ...state, complete: useCreateEndpointFormComplete(state.form.control, 'codex') };
    });
    expect(result.current.complete).toBe(true);
    act(() => result.current.form.setValue('name', ''));
    expect(result.current.complete).toBe(false);
    act(() => result.current.form.setValue('name', 'invalid!'));
    act(() => result.current.handleNameBlur());
    expect(result.current.form.getFieldState('name').error).toBeDefined();
    act(() => {
      result.current.form.clearErrors('name');
      result.current.form.setValue('name', 'taken');
    });
    act(() => result.current.handleNameBlur());
    expect(result.current.form.getFieldState('name').error?.message).toContain('already exists');
    act(() => {
      result.current.form.clearErrors('name');
      result.current.form.setValue('name', 'corrected');
    });
    act(() => result.current.handleNameBlur());
    expect(result.current.form.getFieldState('name').error).toBeUndefined();
    expect(result.current.complete).toBe(true);
  });
});
