import { useMutation, useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { loadIssueDetectionTraceScope, saveIssueDetectionTraceScope } from './issueDetectionTraceScope';
import { ISSUE_DETECTION_SCHEDULE_QUERY_KEY } from './useIssueDetectionSchedule';

export const ISSUE_DETECTION_TRACE_SCOPE_QUERY_KEY = 'EXPERIMENT_ISSUE_DETECTION_TRACE_SCOPE';

export const useIssueDetectionTraceScope = (experimentId: string, enabled = true) => {
  const workspace = useActiveWorkspace();
  const client = useQueryClient();
  const queryKey = [ISSUE_DETECTION_TRACE_SCOPE_QUERY_KEY, workspace, experimentId];
  const query = useQuery({
    queryKey,
    queryFn: () => loadIssueDetectionTraceScope(experimentId),
    enabled: enabled && Boolean(experimentId),
    retry: false,
    refetchOnWindowFocus: false,
  });
  const save = useMutation({
    mutationFn: (scope: Parameters<typeof saveIssueDetectionTraceScope>[1]) =>
      saveIssueDetectionTraceScope(experimentId, scope),
    onMutate: () => client.cancelQueries({ queryKey }),
    onSuccess: (saved) => {
      client.setQueryData(queryKey, saved);
      void client
        .invalidateQueries([ISSUE_DETECTION_SCHEDULE_QUERY_KEY, workspace, experimentId])
        .catch(() => undefined);
    },
  });
  return { ...query, save };
};
