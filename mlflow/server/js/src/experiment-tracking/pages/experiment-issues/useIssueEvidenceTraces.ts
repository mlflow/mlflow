import { useMemo } from 'react';
import { useInfiniteQuery } from '@databricks/web-shared/query-client';
import type { ModelTraceInfoV3 } from '../../../shared/web-shared/model-trace-explorer';
import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';

export const ISSUE_EVIDENCE_TRACES_QUERY_KEY = 'ISSUE_EVIDENCE_TRACES';

/** One page of evidence cards. Small on purpose: each card's spans are read separately. */
export const ISSUE_EVIDENCE_PAGE_SIZE = 5;

/**
 * Issue IDs are `iss-` followed by URL-safe characters, which is also what the store accepts. The
 * ID reaches this hook from the URL, so anything else is not searched rather than interpolated
 * into a filter expression it could otherwise terminate.
 */
const isSearchableIssueId = (issueId: string | undefined): boolean =>
  Boolean(issueId && /^iss-[A-Za-z0-9_-]{1,32}$/.test(issueId));

interface TracesPage {
  traces?: ModelTraceInfoV3[];
  next_page_token?: string;
}

/**
 * The traces an issue affects, newest first, one page at a time.
 *
 * Membership comes from the issue references themselves through the existing `issue.id` trace
 * filter, so the affected population is whatever the store currently associates -- no second
 * copy of the trace list, and a reference added by a later detection run shows up here. Trace
 * *info* only: a header needs the trace's name, age, duration and state, and one expanded card's
 * spans cost orders of magnitude more, so those are read per card when it opens.
 */
export const useIssueEvidenceTraces = (experimentId: string, issueId: string | undefined) => {
  const workspace = useActiveWorkspace();
  const query = useInfiniteQuery<TracesPage, Error>({
    queryKey: [ISSUE_EVIDENCE_TRACES_QUERY_KEY, workspace, experimentId, issueId],
    queryFn: ({ pageParam, signal }) =>
      fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
        method: 'POST',
        signal,
        body: {
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
          filter: `issue.id = '${issueId}'`,
          order_by: ['timestamp DESC'],
          max_results: ISSUE_EVIDENCE_PAGE_SIZE,
          page_token: pageParam,
        },
      }),
    getNextPageParam: (page) => page.next_page_token || undefined,
    enabled: isSearchableIssueId(issueId),
    retry: false,
  });

  const traceInfos = useMemo(() => query.data?.pages.flatMap((page) => page.traces ?? []) ?? [], [query.data]);

  return { ...query, traceInfos };
};
