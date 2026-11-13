import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import type { InfiniteData, QueryKey } from '@databricks/web-shared/query-client';
import { NotFoundError } from '@databricks/web-shared/errors';
import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { GET_ISSUE_QUERY_KEY, getIssue } from '../../components/run-page/hooks/useGetIssueQuery';
import { SEARCH_ISSUES_QUERY_KEY } from '../../components/run-page/hooks/useSearchIssuesQuery';
import type { Issue, IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';

export interface IssueFilters {
  name: string;
  statuses: IssueStatus[];
  severities: IssueSeverity[];
  categories: string[];
  orderBy: string[];
}

interface IssuesPage {
  issues?: Issue[];
  next_page_token?: string;
  total_count?: number;
}

const issueListKey = (workspace: string | null, experimentId: string) =>
  [SEARCH_ISSUES_QUERY_KEY, 'experiment', workspace, experimentId] as const;
const issueDetailKey = (workspace: string | null, experimentId: string, issueId?: string) =>
  [GET_ISSUE_QUERY_KEY, 'experiment', workspace, experimentId, issueId] as const;

export const useExperimentIssues = (experimentId: string, filters: IssueFilters) => {
  const workspace = useActiveWorkspace();
  return useInfiniteQuery<IssuesPage, Error>({
    queryKey: [...issueListKey(workspace, experimentId), filters],
    queryFn: ({ pageParam, signal }) =>
      fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
        method: 'POST',
        signal,
        body: {
          experiment_id: experimentId,
          name: filters.name,
          statuses: filters.statuses,
          severities: filters.severities,
          categories: filters.categories,
          order_by: filters.orderBy,
          max_results: 50,
          page_token: pageParam,
          include_total_count: true,
          include_trace_count: true,
          view: 'basic',
        },
      }),
    getNextPageParam: (page) => page.next_page_token || undefined,
    retry: false,
  });
};

export const useExperimentIssue = (experimentId: string, issueId?: string) => {
  const workspace = useActiveWorkspace();
  return useQuery<Issue, Error>({
    queryKey: issueDetailKey(workspace, experimentId, issueId),
    queryFn: async () => {
      const issue = await getIssue(issueId ?? '');
      if (issue.experiment_id !== experimentId) {
        throw new NotFoundError({ status: 404 });
      }
      return issue;
    },
    enabled: Boolean(issueId),
    retry: false,
  });
};

export const useHasExperimentIssues = (experimentId: string, enabled: boolean) => {
  const workspace = useActiveWorkspace();
  return useQuery<IssuesPage, Error>({
    queryKey: [...issueListKey(workspace, experimentId), 'any'],
    queryFn: ({ signal }) =>
      fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/search'), {
        method: 'POST',
        signal,
        body: { experiment_id: experimentId, max_results: 1, view: 'basic' },
      }),
    enabled,
    retry: false,
  });
};

const replaceIssue = (data: InfiniteData<IssuesPage> | undefined, issueId: string, patch: Partial<Issue>) =>
  data && {
    ...data,
    pages: data.pages.map((page) => ({
      ...page,
      issues: page.issues?.map((issue) => (issue.issue_id === issueId ? { ...issue, ...patch } : issue)),
    })),
  };

export const useTriageExperimentIssue = (experimentId: string) => {
  const queryClient = useQueryClient();
  const workspace = useActiveWorkspace();
  const listKey = issueListKey(workspace, experimentId);
  return useMutation({
    mutationFn: ({ issueId, status }: { issueId: string; status: IssueStatus }) =>
      fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/issues/${encodeURIComponent(issueId)}`), {
        method: 'PATCH',
        body: { issue_id: issueId, status },
      }) as Promise<{ issue: Issue }>,
    onMutate: async ({ issueId, status }) => {
      const detailKey = issueDetailKey(workspace, experimentId, issueId);
      await Promise.all([queryClient.cancelQueries(listKey), queryClient.cancelQueries(detailKey)]);
      const previousDetail = queryClient.getQueryData<Issue>(detailKey);
      const previousRows: [QueryKey, Issue][] = [];
      for (const [key, data] of queryClient.getQueriesData<InfiniteData<IssuesPage>>({ queryKey: listKey })) {
        const row = data?.pages?.flatMap((page) => page.issues ?? []).find((issue) => issue.issue_id === issueId);
        if (row) {
          previousRows.push([key, row]);
          queryClient.setQueryData(key, replaceIssue(data, issueId, { status }));
        }
      }
      if (previousDetail) queryClient.setQueryData(detailKey, { ...previousDetail, status });
      return { previousDetail, previousRows, detailKey };
    },
    onError: (_error, { issueId }, context) => {
      if (context?.previousDetail) queryClient.setQueryData(context.detailKey, context.previousDetail);
      // Restore only this issue; another issue may have been updated while the request was pending.
      context?.previousRows.forEach(([key, row]) =>
        queryClient.setQueryData<InfiniteData<IssuesPage>>(key, (data) => replaceIssue(data, issueId, row)),
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries([SEARCH_ISSUES_QUERY_KEY]);
      queryClient.invalidateQueries([GET_ISSUE_QUERY_KEY]);
    },
  });
};
