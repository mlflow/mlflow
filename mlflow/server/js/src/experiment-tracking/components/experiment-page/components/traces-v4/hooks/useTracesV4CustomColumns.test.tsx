import { describe, expect, it } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';

import { useTracesV4CustomColumns } from './useTracesV4CustomColumns';

const trace = (over: Partial<ModelTraceInfoV3>): ModelTraceInfoV3 =>
  ({ tags: {}, trace_metadata: {}, ...over }) as ModelTraceInfoV3;

const wrapper = ({ children }: { children: React.ReactNode }) => <IntlProvider locale="en">{children}</IntlProvider>;

const useCustomColumns = (traces: ModelTraceInfoV3[]) => {
  const [visibility, setVisibility] = useState<Record<string, boolean>>({});
  return useTracesV4CustomColumns(traces, visibility, setVisibility);
};

describe('useTracesV4CustomColumns', () => {
  it('offers user tags and metadata in separate opt-in groups', () => {
    const { result } = renderHook(
      () =>
        useCustomColumns([
          trace({
            tags: { environment: 'prod', 'mlflow.internal': 'hidden' },
            trace_metadata: {
              region: 'us-west',
              'mlflow.trace.tokenUsage': '{}',
            },
          }),
        ]),
      { wrapper },
    );

    expect(result.current.tags.selectorOptions.map((opt) => opt.id)).toEqual(['tag:environment']);
    expect(result.current.metadata.selectorOptions.map((opt) => opt.id)).toEqual(['custom_metadata:region']);
    expect(result.current.columnDefs).toEqual([]);
  });

  it('renders an opted-in tag and metadata field as table columns', () => {
    const traces = [
      trace({
        tags: { environment: 'prod' },
        trace_metadata: { region: 'us-west' },
      }),
    ];
    const { result } = renderHook(() => useCustomColumns(traces), { wrapper });

    act(() => {
      result.current.tags.toggle('tag:environment');
      result.current.metadata.toggle('custom_metadata:region');
    });

    expect(result.current.columnDefs.map((col) => col.id)).toEqual(['tag:environment', 'custom_metadata:region']);
    expect(result.current.tags.visibleIds).toEqual(['tag:environment']);
    expect(result.current.metadata.visibleIds).toEqual(['custom_metadata:region']);
  });

  it('keeps an opted-in field available when it is absent from a later page', () => {
    const { result, rerender } = renderHook(({ traces }: { traces: ModelTraceInfoV3[] }) => useCustomColumns(traces), {
      initialProps: { traces: [trace({ tags: { environment: 'prod' } })] },
      wrapper,
    });
    act(() => result.current.tags.toggle('tag:environment'));

    rerender({ traces: [trace({})] });

    expect(result.current.tags.selectorOptions.map((opt) => opt.id)).toEqual(['tag:environment']);
    expect(result.current.tags.visibleIds).toEqual(['tag:environment']);
    expect(result.current.columnDefs.map((col) => col.id)).toEqual(['tag:environment']);
  });
});
