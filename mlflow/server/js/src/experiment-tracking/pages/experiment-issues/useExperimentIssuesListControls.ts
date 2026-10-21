import { useCallback, useMemo } from 'react';
import { useSearchParams } from '../../../common/utils/RoutingUtils';
import { ISSUE_CATEGORY_DEFINITIONS } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import type { IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { useDebouncedSearchInput } from '../experiment-evaluation-datasets-v2/hooks/useDebouncedSearchInput';
import type { IssueFilters } from './useExperimentIssues';

export const ISSUE_DETECTION_CATEGORIES = ISSUE_CATEGORY_DEFINITIONS.map(({ id }) => id);
export const ISSUE_DETECTION_SEVERITIES: IssueSeverity[] = ['high', 'medium', 'low'];
export const ISSUE_DETECTION_TRIAGE_STATUSES: IssueStatus[] = ['pending', 'resolved', 'rejected'];
export const BACKEND_ISSUE_SORT_KEYS = ['severity', 'mostRecent', 'lastSeen', 'lastUpdated'] as const;
export type IssueSortKey = (typeof BACKEND_ISSUE_SORT_KEYS)[number];
export const isIssueSortKey = (value: string | null): value is IssueSortKey =>
  value !== null && BACKEND_ISSUE_SORT_KEYS.some((key) => key === value);
const ISSUE_SEVERITY_PARAM = 'issueSeverity';
const ISSUE_CATEGORY_PARAM = 'issueCategory';
const ISSUE_SORT_PARAM = 'issueSort';
const ISSUE_STATUS_PARAM = 'issueStatus';
const ISSUE_SEARCH_PARAM = 'issueSearch';

export interface ExperimentIssuesListState {
  sortKey: IssueSortKey;
  setSortKey: (key: IssueSortKey) => void;
  severities: IssueSeverity[];
  setSeverities: (values: IssueSeverity[]) => void;
  categories: string[];
  setCategories: (values: string[]) => void;
  statuses: IssueStatus[];
  setStatuses: (values: IssueStatus[]) => void;
  hasActiveFilters: boolean;
  clearFilters: () => void;
}
export type ExperimentIssuesToolbarControls = ExperimentIssuesListState;

const SORTS: Record<IssueSortKey, string[]> = {
  severity: ['status ASC', 'severity DESC', 'last_seen_timestamp DESC'],
  lastSeen: ['last_seen_timestamp DESC'],
  mostRecent: ['created_timestamp DESC'],
  lastUpdated: ['last_updated_timestamp DESC'],
};

export const useExperimentIssuesListState = (): ExperimentIssuesListState => {
  const [params, setSearchParams] = useSearchParams();
  const severityParams = params.getAll(ISSUE_SEVERITY_PARAM);
  const categoryParams = params.getAll(ISSUE_CATEGORY_PARAM);
  const sortParam = params.get(ISSUE_SORT_PARAM);
  const statusParams = params.getAll(ISSUE_STATUS_PARAM);

  // Filtering the canonical enums both validates, deduplicates, and gives shared links a stable order.
  const severities = useMemo(
    () => ISSUE_DETECTION_SEVERITIES.filter((severity) => severityParams.includes(severity)),
    [severityParams],
  );
  const categories = useMemo(() => Array.from(new Set(categoryParams)), [categoryParams]);
  const statuses = useMemo<IssueStatus[]>(() => {
    if (statusParams.length === 0) {
      return ['pending'];
    }
    if (statusParams.includes('all')) {
      return [];
    }
    const validStatuses = ISSUE_DETECTION_TRIAGE_STATUSES.filter((status) => statusParams.includes(status));
    return validStatuses.length > 0 ? validStatuses : ['pending'];
  }, [statusParams]);
  const sortKey: IssueSortKey = isIssueSortKey(sortParam)
    ? sortParam
    : sortParam === 'newest'
      ? 'mostRecent'
      : sortParam === 'updated'
        ? 'lastUpdated'
        : 'severity';

  const setSeverities = useCallback(
    (values: IssueSeverity[]) => {
      setSearchParams(
        (params) => {
          params.delete(ISSUE_SEVERITY_PARAM);
          for (const value of ISSUE_DETECTION_SEVERITIES) {
            if (values.includes(value)) {
              params.append(ISSUE_SEVERITY_PARAM, value);
            }
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setCategories = useCallback(
    (values: string[]) => {
      setSearchParams(
        (params) => {
          params.delete(ISSUE_CATEGORY_PARAM);
          for (const value of new Set(values)) {
            params.append(ISSUE_CATEGORY_PARAM, value);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setSortKey = useCallback(
    (key: IssueSortKey) => {
      setSearchParams(
        (params) => {
          if (key === 'severity') {
            params.delete(ISSUE_SORT_PARAM);
          } else {
            params.set(ISSUE_SORT_PARAM, key);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setStatuses = useCallback(
    (values: IssueStatus[]) => {
      setSearchParams(
        (params) => {
          params.delete(ISSUE_STATUS_PARAM);
          const validStatuses = ISSUE_DETECTION_TRIAGE_STATUSES.filter((status) => values.includes(status));
          if (validStatuses.length === 0) {
            params.append(ISSUE_STATUS_PARAM, 'all');
          } else if (!(validStatuses.length === 1 && validStatuses[0] === 'pending')) {
            for (const status of validStatuses) {
              params.append(ISSUE_STATUS_PARAM, status);
            }
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const clearFilters = useCallback(() => {
    setSearchParams(
      (params) => {
        params.delete(ISSUE_SEVERITY_PARAM);
        params.delete(ISSUE_CATEGORY_PARAM);
        params.delete(ISSUE_STATUS_PARAM);
        params.append(ISSUE_STATUS_PARAM, 'all');
        return params;
      },
      { replace: true },
    );
  }, [setSearchParams]);

  return {
    sortKey,
    setSortKey,
    severities,
    setSeverities,
    categories,
    setCategories,
    statuses,
    setStatuses,
    hasActiveFilters: severities.length > 0 || categories.length > 0 || statuses.length > 0,
    clearFilters,
  };
};

export const useIssueListControls = () => {
  const state = useExperimentIssuesListState();
  const [params, setParams] = useSearchParams();
  const setSearch = useCallback(
    (value: string) =>
      setParams(
        (current) => {
          if (value) current.set(ISSUE_SEARCH_PARAM, value);
          else current.delete(ISSUE_SEARCH_PARAM);
          return current;
        },
        { replace: true },
      ),
    [setParams],
  );
  const search = useDebouncedSearchInput({
    committedValue: params.get(ISSUE_SEARCH_PARAM) ?? '',
    onCommit: setSearch,
    debounceMs: 300,
  });
  const filters: IssueFilters = {
    name: params.get(ISSUE_SEARCH_PARAM) ?? '',
    statuses: state.statuses,
    severities: state.severities,
    categories: state.categories.flatMap((category) =>
      ISSUE_DETECTION_CATEGORIES.some((id) => id === category) ? [category, `[${category}]`] : [category],
    ),
    orderBy: SORTS[state.sortKey],
  };
  return { ...state, search, filters };
};
