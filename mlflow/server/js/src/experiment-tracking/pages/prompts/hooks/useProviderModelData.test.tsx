import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { useProviderModelData } from './useProviderModelData';
import { useProvidersQuery } from '../../../../gateway/hooks/useProvidersQuery';
import { useModelsQuery } from '../../../../gateway/hooks/useModelsQuery';

// Mock the gateway hooks
jest.mock('../../../../gateway/hooks/useProvidersQuery');
jest.mock('../../../../gateway/hooks/useModelsQuery');

const mockUseProvidersQuery = useProvidersQuery as jest.MockedFunction<typeof useProvidersQuery>;
const mockUseModelsQuery = useModelsQuery as jest.MockedFunction<typeof useModelsQuery>;

describe('useProviderModelData', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return providers in priority order', () => {
    const mockProviders = ['azure', 'openai', 'grok', 'anthropic', 'gemini', 'databricks'];
    mockUseProvidersQuery.mockReturnValue({
      data: mockProviders,
      isLoading: false,
    } as any);
    mockUseModelsQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
    } as any);

    const onProviderChange = jest.fn();
    const { result } = renderHook(() => useProviderModelData(undefined, onProviderChange));

    // Priority providers should come first: openai, anthropic, gemini, databricks
    // Then others alphabetically: azure, bedrock
    expect(result.current.providers).toEqual(['openai', 'anthropic', 'gemini', 'databricks', 'azure', 'grok']);
  });

  it('should return undefined providers when data is not loaded', () => {
    mockUseProvidersQuery.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);
    mockUseModelsQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
    } as any);

    const onProviderChange = jest.fn();
    const { result } = renderHook(() => useProviderModelData(undefined, onProviderChange));

    expect(result.current.providers).toBeUndefined();
    expect(result.current.providersLoading).toBe(true);
  });

  it('should fetch models for selected provider', () => {
    mockUseProvidersQuery.mockReturnValue({
      data: ['openai'],
      isLoading: false,
    } as any);
    mockUseModelsQuery.mockReturnValue({
      data: [{ model: 'gpt-4' }, { model: 'gpt-3.5-turbo' }],
      isLoading: false,
    } as any);

    const onProviderChange = jest.fn();
    const { result } = renderHook(() => useProviderModelData('openai', onProviderChange));

    expect(result.current.models).toEqual([{ model: 'gpt-4' }, { model: 'gpt-3.5-turbo' }]);
    expect(result.current.modelsLoading).toBe(false);
  });

  it('should call onProviderChange when provider changes', async () => {
    mockUseProvidersQuery.mockReturnValue({
      data: ['openai', 'anthropic'],
      isLoading: false,
    } as any);
    mockUseModelsQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
    } as any);

    const onProviderChange = jest.fn();
    const { rerender } = renderHook(({ provider }) => useProviderModelData(provider, onProviderChange), {
      initialProps: { provider: 'openai' },
    });

    // Change provider
    rerender({ provider: 'anthropic' });

    await waitFor(() => {
      expect(onProviderChange).toHaveBeenCalledTimes(1);
    });
  });

  it('should not call onProviderChange when provider stays the same', async () => {
    mockUseProvidersQuery.mockReturnValue({
      data: ['openai'],
      isLoading: false,
    } as any);
    mockUseModelsQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
    } as any);

    const onProviderChange = jest.fn();
    const { rerender } = renderHook(({ provider }) => useProviderModelData(provider, onProviderChange), {
      initialProps: { provider: 'openai' },
    });

    // Rerender with same provider
    rerender({ provider: 'openai' });

    await waitFor(() => {
      expect(onProviderChange).not.toHaveBeenCalled();
    });
  });

  it('should handle loading states correctly', () => {
    mockUseProvidersQuery.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);
    mockUseModelsQuery.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as any);

    const onProviderChange = jest.fn();
    const { result } = renderHook(() => useProviderModelData('openai', onProviderChange));

    expect(result.current.providersLoading).toBe(true);
    expect(result.current.modelsLoading).toBe(true);
  });
});
