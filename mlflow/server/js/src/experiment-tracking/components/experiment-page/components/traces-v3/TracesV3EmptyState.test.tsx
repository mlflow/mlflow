import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { render, screen } from '@testing-library/react';

import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';
import { useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';

import { TracesV3EmptyState } from './TracesV3EmptyState';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { useMonitoringFilters } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';

jest.mock('@databricks/web-shared/genai-traces-table', () => {
  const actual = jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  );
  return {
    ...actual,
    useSearchMlflowTraces: jest.fn(),
  };
});

jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig', () => ({
  useMonitoringConfig: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters', () => ({
  useMonitoringFilters: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/utils/ExperimentKindUtils', () => ({
  useExperimentKind: () => undefined,
  isGenAIExperimentKind: () => false,
}));

jest.mock('../../../traces/quickstart/TracesViewTableNoTracesQuickstart', () => ({
  TracesViewTableNoTracesQuickstart: () => <div data-testid="quickstart" />,
}));

const refresh = jest.fn();

const mockProbe = (overrides: { data?: unknown[]; isFetching?: boolean; isLoading?: boolean } = {}) => {
  jest.mocked(useSearchMlflowTraces).mockReturnValue({
    data: (overrides.data ?? []) as never[],
    isLoading: overrides.isLoading ?? false,
    isFetching: overrides.isFetching ?? false,
  });
};

const lastRefetchInterval = () => {
  const [{ refetchInterval }] = jest.mocked(useSearchMlflowTraces).mock.calls.at(-1)!;
  return refetchInterval;
};

// Keep providers stable across rerenders so the TracesV3EmptyState instance —
// and therefore `hasSeenTrace` state and the initial-fetch ref — persists.
// We build fresh JSX each render so React doesn't bail on identical elements.
const renderEmptyState = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const buildUi = () => (
    <IntlProvider locale="en">
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <TracesV3EmptyState experimentIds={['exp-1']} traceSearchLocations={[]} />
        </DesignSystemProvider>
      </QueryClientProvider>
    </IntlProvider>
  );
  const result = render(buildUi());
  return { ...result, rerender: () => result.rerender(buildUi()) };
};

describe('TracesV3EmptyState', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    refresh.mockClear();
    jest.mocked(useMonitoringConfig).mockReturnValue({
      dateNow: new Date('2026-01-01T00:00:00Z'),
      lastRefreshTime: 0,
      refresh,
    });
    jest
      .mocked(useMonitoringFilters)
      .mockReturnValue([
        { startTimeLabel: 'LAST_7_DAYS', startTime: undefined, endTime: undefined },
        jest.fn(),
        false,
      ] as never);
    jest.mocked(useGetExperimentQuery).mockReturnValue({
      data: { name: 'Default', experimentId: 'exp-1', tags: [] },
      loading: false,
    } as never);
  });

  it('renders the quickstart and polls at the interval when the probe finds no traces', () => {
    mockProbe({ data: [], isFetching: false });

    renderEmptyState();

    expect(screen.getByTestId('quickstart')).toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
    expect(typeof lastRefetchInterval()).toBe('number');
    expect(lastRefetchInterval()).toBeGreaterThan(0);
  });

  it('does not fire refresh on stale cached data while a refetch is in flight (initial-fetch latch)', () => {
    // Simulates remount after a delete where the probe's cache still holds
    // stale traces and `keepPreviousData` shows them while refetching.
    mockProbe({ data: [{ trace_id: 'stale' }], isFetching: true });

    renderEmptyState();

    expect(refresh).not.toHaveBeenCalled();
    expect(screen.getByTestId('quickstart')).toBeInTheDocument();
  });

  it('transitions from the quickstart to the filter-hidden empty state when the probe finds a trace', () => {
    mockProbe({ data: [], isFetching: false });
    const { rerender } = renderEmptyState();
    expect(screen.getByTestId('quickstart')).toBeInTheDocument();

    mockProbe({ data: [{ trace_id: 'fresh' }], isFetching: false });
    rerender();

    expect(screen.queryByTestId('quickstart')).not.toBeInTheDocument();
    expect(screen.getByText('No traces found')).toBeInTheDocument();
  });

  it('flips refetchInterval to false once the probe detects a trace (hasSeenTrace latches)', () => {
    mockProbe({ data: [], isFetching: false });
    const { rerender } = renderEmptyState();
    expect(typeof lastRefetchInterval()).toBe('number');

    mockProbe({ data: [{ trace_id: 'fresh' }], isFetching: false });
    rerender();

    expect(lastRefetchInterval()).toBe(false);
  });

  it('fires refresh exactly once when a trace first appears, even across subsequent rerenders', () => {
    mockProbe({ data: [], isFetching: false });
    const { rerender } = renderEmptyState();
    expect(refresh).not.toHaveBeenCalled();

    // Trace appears: refresh should fire once.
    mockProbe({ data: [{ trace_id: 'fresh' }], isFetching: false });
    rerender();
    expect(refresh).toHaveBeenCalledTimes(1);

    // Additional rerenders with traces still present must not refire refresh.
    mockProbe({ data: [{ trace_id: 'fresh' }, { trace_id: 'fresh2' }], isFetching: false });
    rerender();
    rerender();
    expect(refresh).toHaveBeenCalledTimes(1);
  });
});
