import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { useEffect, useState } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { getLocalStorageItem, setLocalStorageItem } from '@databricks/web-shared/hooks';

import { useLocation, useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { EvalRunsFilterOperator } from '../evalRunsFilterFields';
import type { EvalRunsFilterClause } from '../evalRunsFilterFields';
import { DATE_PRESET_ORDER } from './useEvalRunsFilterLabels';
import {
  EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM,
  EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM,
  EvalRunsDatePreset,
  useEvalRunsFilters,
} from './useEvalRunsFilters';

const EXPERIMENT_ID = 'exp-1';
const DATE_PRESET_STORAGE_VERSION = 1;
const datePresetStorageKey = (experimentId: string) => `mlflow.eval-runs.date-preset.${experimentId}`;

const runNameClause: EvalRunsFilterClause = {
  field: 'attributes.run_name',
  operator: EvalRunsFilterOperator.EQUALS,
  value: 'foo',
};

const datasetClause = (name: string): EvalRunsFilterClause => ({
  field: 'dataset.name',
  operator: EvalRunsFilterOperator.EQUALS,
  value: name,
});

type FilterHookProps = Omit<Parameters<typeof useEvalRunsFilters>[0], 'experimentId'>;

describe('useEvalRunsFilters', () => {
  let navigateInTest: ((url: string) => void) | undefined;
  const history = { replace: (url: string) => navigateInTest?.(url) };
  const NavigateSpy = () => {
    const navigate = useNavigate();
    navigateInTest = (url) => navigate(url, { replace: true });
    return null;
  };

  let lastSearch = '';
  const LocationSpy = () => {
    const { search } = useLocation();
    useEffect(() => {
      lastSearch = search;
    }, [search]);
    return null;
  };

  const mountHook = async (props: FilterHookProps, initialUrl = '/p') => {
    lastSearch = '';
    act(() => history.replace(initialUrl));
    const rendered = renderHook(
      () => {
        const [experimentId, setExperimentId] = useState(EXPERIMENT_ID);
        return { ...useEvalRunsFilters({ ...props, experimentId }), setExperimentId };
      },
      {
        wrapper: ({ children }: { children: React.ReactNode }) => (
          <TestRouter
            initialEntries={[initialUrl]}
            routes={[
              testRoute(
                <>
                  <LocationSpy />
                  <NavigateSpy />
                  <div>{children}</div>
                </>,
              ),
            ]}
          />
        ),
      },
    );
    await waitForRoutesToBeRendered();
    await waitFor(() => expect(rendered.result.current).not.toBeNull());
    return rendered;
  };

  const param = (key: string) => new URLSearchParams(lastSearch).get(key);
  const urlWithParams = (params: Record<string, string>) => `/p?${new URLSearchParams(params)}`;

  afterEach(() => {
    jest.clearAllMocks();
    window.localStorage.clear();
  });

  // @ui-test-skill-generated
  it('composes the search text, the date-preset lower bound, and the compiled clauses into one AND-joined string', async () => {
    const { result } = await mountHook(
      { fallbackDatePreset: EvalRunsDatePreset.LAST_7_DAYS },
      urlWithParams({ [EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM]: 'run' }),
    );

    // Search + date bound, before any clause is added. (The bound is `now - window`, so it may be
    // negative under the test env's pinned clock.)
    expect(result.current.filterString).toMatch(/^run AND attributes\.start_time > -?\d+$/);

    act(() => result.current.setClauses([runNameClause]));

    // The clause lands after the date bound, AND-joined.
    expect(result.current.filterString).toMatch(
      /^run AND attributes\.start_time > -?\d+ AND attributes\.run_name = 'foo'$/,
    );
  });

  it('reports the filter active when the date preset is not ALL and inactive when it is ALL with no clauses', async () => {
    const { result } = await mountHook({
      fallbackDatePreset: EvalRunsDatePreset.ALL,
      persistDatePreset: true,
    });

    expect(result.current.isAnyFilterActive).toBe(false);

    act(() => result.current.setDatePreset(EvalRunsDatePreset.LAST_HOUR));

    expect(result.current.isAnyFilterActive).toBe(true);
  });

  it('reports the filter active when a clause is set even at the ALL date preset', async () => {
    const { result } = await mountHook({ fallbackDatePreset: EvalRunsDatePreset.ALL });

    expect(result.current.isAnyFilterActive).toBe(false);

    act(() => result.current.setClauses([runNameClause]));

    expect(result.current.isAnyFilterActive).toBe(true);
  });

  it('reports the filter active when search text is set at the ALL date preset', async () => {
    const { result } = await mountHook(
      { fallbackDatePreset: EvalRunsDatePreset.ALL },
      urlWithParams({ [EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM]: 'run' }),
    );

    expect(result.current.isAnyFilterActive).toBe(true);
  });

  it('preserves clauses when the date preset changes and preserves the date preset when clauses change', async () => {
    const { result } = await mountHook({
      fallbackDatePreset: EvalRunsDatePreset.LAST_7_DAYS,
      persistDatePreset: true,
    });

    act(() => result.current.setClauses([runNameClause]));
    // Setting clauses leaves the date preset untouched.
    expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_7_DAYS);
    expect(result.current.filters.clauses).toEqual([runNameClause]);

    act(() => result.current.setDatePreset(EvalRunsDatePreset.ALL));
    // Changing the date preset leaves the clauses untouched; the compiled string keeps the clause and drops the bound.
    expect(result.current.filters.clauses).toEqual([runNameClause]);
    expect(result.current.filterString).toBe("attributes.run_name = 'foo'");
    expect(result.current.filterString).not.toContain('attributes.start_time');
  });

  it('produces an empty filter string at the ALL preset with no search text and no clauses', async () => {
    const { result } = await mountHook({ fallbackDatePreset: EvalRunsDatePreset.ALL });

    expect(result.current.filterString).toBe('');
  });

  it('restores search and structured filters from the URL', async () => {
    const { result } = await mountHook(
      { fallbackDatePreset: EvalRunsDatePreset.ALL },
      urlWithParams({
        [EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM]: "attributes.status = 'FAILED'",
        [EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM]: JSON.stringify([runNameClause]),
      }),
    );

    expect(result.current.searchFilter).toBe("attributes.status = 'FAILED'");
    expect(result.current.filters.clauses).toEqual([runNameClause]);
    expect(result.current.filterString).toBe("attributes.status = 'FAILED' AND attributes.run_name = 'foo'");
  });

  it('updates URL filters without dropping run and opened-trace state', async () => {
    const { result } = await mountHook(
      { fallbackDatePreset: EvalRunsDatePreset.ALL },
      '/p?selectedRunUuid=run-1&selectedEvaluationId=trace-1',
    );

    act(() => result.current.setSearchFilter((current: string) => `${current}attributes.status = 'FAILED'`));
    act(() => result.current.setClauses([runNameClause]));

    expect(param('selectedRunUuid')).toBe('run-1');
    expect(param('selectedEvaluationId')).toBe('trace-1');
    expect(param(EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM)).toBe("attributes.status = 'FAILED'");
    expect(JSON.parse(param(EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM) ?? '')).toEqual([runNameClause]);
    expect(param('searchFilter')).toBeNull();
    expect(param('filterClauses')).toBeNull();
  });

  it('removes empty URL filters and ignores malformed structured filters', async () => {
    const { result } = await mountHook(
      { fallbackDatePreset: EvalRunsDatePreset.ALL },
      urlWithParams({
        [EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM]: 'run',
        [EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM]: '{invalid-json',
      }),
    );

    expect(result.current.filters.clauses).toEqual([]);
    act(() => result.current.setSearchFilter(''));
    act(() => result.current.setClauses([]));

    expect(param(EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM)).toBeNull();
    expect(param(EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM)).toBeNull();
  });

  // @ui-test-skill-generated
  it('replaces the dataset clause while preserving the date preset and other clauses', async () => {
    const { result } = await mountHook(
      { fallbackDatePreset: EvalRunsDatePreset.LAST_7_DAYS },
      urlWithParams({ [EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM]: 'quality >= 0.8' }),
    );

    act(() => result.current.setClauses([runNameClause, datasetClause('baseline')]));
    expect(result.current.selectedDataset).toBe('baseline');
    expect(result.current.hasDatasetFilter).toBe(true);

    act(() => result.current.setSelectedDataset('candidate'));

    expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_7_DAYS);
    expect(result.current.filters.clauses).toEqual([runNameClause, datasetClause('candidate')]);
    expect(result.current.filterString).toContain('quality >= 0.8');
    expect(result.current.filterString).toContain('attributes.start_time >');
    expect(result.current.filterString).toContain("attributes.run_name = 'foo'");
    expect(result.current.filterString).toContain("dataset.name = 'candidate'");
    expect(result.current.filterString).not.toContain("dataset.name = 'baseline'");
  });

  it('uses the latest clauses when dataset selection follows another update in the same tick', async () => {
    const { result } = await mountHook({ fallbackDatePreset: EvalRunsDatePreset.ALL });

    act(() => {
      result.current.setClauses([runNameClause]);
      result.current.setSelectedDataset('candidate');
    });

    expect(result.current.filters.clauses).toEqual([runNameClause, datasetClause('candidate')]);
  });

  // @ui-test-skill-generated
  it('clears only the dataset clause and keeps non-dataset filters active', async () => {
    const { result } = await mountHook({
      fallbackDatePreset: EvalRunsDatePreset.LAST_30_DAYS,
    });

    act(() => result.current.setClauses([runNameClause, datasetClause('candidate')]));
    act(() => result.current.setSelectedDataset(undefined));

    expect(result.current.selectedDataset).toBeUndefined();
    expect(result.current.hasDatasetFilter).toBe(false);
    expect(result.current.filters.clauses).toEqual([runNameClause]);
    expect(result.current.filterString).toContain('attributes.start_time >');
    expect(result.current.filterString).toContain("attributes.run_name = 'foo'");
    expect(result.current.filterString).not.toContain('dataset.name');
  });

  // @ui-test-skill-generated
  it('keeps a non-equality dataset filter active without exposing it as a picker selection', async () => {
    const { result } = await mountHook({
      fallbackDatePreset: EvalRunsDatePreset.ALL,
    });

    act(() =>
      result.current.setClauses([
        {
          field: 'dataset.name',
          operator: EvalRunsFilterOperator.CONTAINS,
          value: 'candidate',
        },
      ]),
    );

    expect(result.current.selectedDataset).toBeUndefined();
    expect(result.current.hasDatasetFilter).toBe(true);
    expect(result.current.filterString).toBe("dataset.name LIKE '%candidate%'");
  });

  it('includes every persisted date preset in the dropdown exactly once', () => {
    expect([...DATE_PRESET_ORDER].sort()).toEqual(Object.values(EvalRunsDatePreset).sort());
  });

  describe('when persistence is off (V1 default)', () => {
    it('keeps the default date preset and does not persist updates', async () => {
      const { result } = await mountHook({ fallbackDatePreset: EvalRunsDatePreset.ALL });

      act(() => result.current.setDatePreset(EvalRunsDatePreset.LAST_HOUR));

      expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.ALL);
      expect(param('startTimeLabel')).toBeNull();
      expect(
        getLocalStorageItem(datePresetStorageKey(EXPERIMENT_ID), DATE_PRESET_STORAGE_VERSION, true, undefined),
      ).toBeUndefined();
    });

    it('ignores persisted date presets', async () => {
      setLocalStorageItem(
        datePresetStorageKey(EXPERIMENT_ID),
        DATE_PRESET_STORAGE_VERSION,
        true,
        EvalRunsDatePreset.LAST_HOUR,
      );
      const { result } = await mountHook(
        { fallbackDatePreset: EvalRunsDatePreset.ALL },
        '/p?startTimeLabel=LAST_30_DAYS',
      );

      expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.ALL);
    });
  });

  describe('when persistence is on (V2)', () => {
    const persistedProps: FilterHookProps = {
      fallbackDatePreset: EvalRunsDatePreset.LAST_7_DAYS,
      persistDatePreset: true,
    };

    it('reads the date preset from the startTimeLabel URL param (deep link)', async () => {
      const { result } = await mountHook(persistedProps, '/p?startTimeLabel=LAST_30_DAYS');

      expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_30_DAYS);
    });

    it('falls back to the default for a bare URL without writing it to the URL', async () => {
      const { result } = await mountHook(persistedProps);

      expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_7_DAYS);
      expect(param('startTimeLabel')).toBeNull();
    });

    it('falls back to the default when the URL label is not a valid preset', async () => {
      const { result } = await mountHook(persistedProps, '/p?startTimeLabel=NOT_A_PRESET');

      expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_7_DAYS);
    });

    it('setDatePreset writes the preset to the URL', async () => {
      const { result } = await mountHook(persistedProps);

      act(() => result.current.setDatePreset(EvalRunsDatePreset.LAST_HOUR));

      expect(param('startTimeLabel')).toBe(EvalRunsDatePreset.LAST_HOUR);
      expect(result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_HOUR);
    });

    it('removes explicit bounds when writing a preset to the URL', async () => {
      const { result } = await mountHook(
        persistedProps,
        '/p?startTimeLabel=CUSTOM&startTime=2025-01-01&endTime=2025-01-02',
      );

      act(() => result.current.setDatePreset(EvalRunsDatePreset.LAST_HOUR));

      expect(param('startTimeLabel')).toBe(EvalRunsDatePreset.LAST_HOUR);
      expect(param('startTime')).toBeNull();
      expect(param('endTime')).toBeNull();
    });

    it('restores a previously chosen preset from localStorage when the URL is bare', async () => {
      const first = await mountHook(persistedProps);
      act(() => first.result.current.setDatePreset(EvalRunsDatePreset.LAST_30_DAYS));
      expect(param('startTimeLabel')).toBe(EvalRunsDatePreset.LAST_30_DAYS);

      const second = await mountHook(persistedProps);
      expect(second.result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_30_DAYS);
    });

    it('loads the preset for the current experiment when the experiment changes', async () => {
      setLocalStorageItem(
        datePresetStorageKey(EXPERIMENT_ID),
        DATE_PRESET_STORAGE_VERSION,
        true,
        EvalRunsDatePreset.LAST_30_DAYS,
      );
      const rendered = await mountHook(persistedProps);

      act(() => history.replace('/p?startTimeLabel=LAST_HOUR'));
      await waitFor(() => expect(rendered.result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_HOUR));
      act(() => history.replace('/p'));
      await waitFor(() => expect(rendered.result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_30_DAYS));
      expect(
        getLocalStorageItem(datePresetStorageKey('exp-2'), DATE_PRESET_STORAGE_VERSION, true, undefined),
      ).toBeUndefined();
      act(() => rendered.result.current.setExperimentId('exp-2'));
      expect(rendered.result.current.filters.datePreset).toBe(EvalRunsDatePreset.LAST_7_DAYS);

      act(() => rendered.result.current.setDatePreset(EvalRunsDatePreset.LAST_HOUR));
      expect(
        getLocalStorageItem(datePresetStorageKey(EXPERIMENT_ID), DATE_PRESET_STORAGE_VERSION, true, undefined),
      ).toBe(EvalRunsDatePreset.LAST_30_DAYS);
      expect(getLocalStorageItem(datePresetStorageKey('exp-2'), DATE_PRESET_STORAGE_VERSION, true, undefined)).toBe(
        EvalRunsDatePreset.LAST_HOUR,
      );
    });
  });
});
