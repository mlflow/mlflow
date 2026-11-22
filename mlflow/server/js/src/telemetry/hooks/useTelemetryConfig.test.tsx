import { describe, expect, it, jest } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '../../common/utils/reactQueryHooks';
import { fetchAPI } from '../../common/utils/FetchUtils';
import { useTelemetryConfig } from './useTelemetryConfig';

jest.mock('../../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

describe('useTelemetryConfig', () => {
  const createWrapper = () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };

  it.each([true, false])('fetches server telemetry disable setting %s', async (disabled) => {
    jest.mocked(fetchAPI).mockResolvedValue({ disable_ui_telemetry: disabled });
    const { result } = renderHook(useTelemetryConfig, { wrapper: createWrapper() });

    expect(result.current.data).toBeUndefined();
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.disable_ui_telemetry).toBe(disabled);
    expect(fetchAPI).toHaveBeenCalledWith('ajax-api/3.0/mlflow/ui-telemetry');
  });

  it('leaves configuration unavailable when the request fails', async () => {
    jest.mocked(fetchAPI).mockRejectedValue(new Error('Unavailable'));
    const { result } = renderHook(useTelemetryConfig, { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.data).toBeUndefined();
  });
});
