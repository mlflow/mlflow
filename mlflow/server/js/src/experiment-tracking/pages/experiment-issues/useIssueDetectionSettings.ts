import { useMutation, useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import {
  loadIssueDetectionSettings,
  saveIssueDetectionSettings,
  type IssueDetectionSettings,
} from './issueDetectionSettings';

export const ISSUE_DETECTION_SETTINGS_QUERY_KEY = 'EXPERIMENT_ISSUE_DETECTION_SETTINGS';

/**
 * The experiment's detection settings, and a way to change them.
 *
 * Shared by the settings modal and by the detection modal that has to send them, so a run
 * submitted right after a save uses what was saved rather than a cached earlier value.
 */
export const useIssueDetectionSettings = (experimentId: string, enabled = true) => {
  const workspace = useActiveWorkspace();
  const queryClient = useQueryClient();
  const queryKey = [ISSUE_DETECTION_SETTINGS_QUERY_KEY, workspace, experimentId];

  const query = useQuery<IssueDetectionSettings, Error>({
    queryKey,
    queryFn: () => loadIssueDetectionSettings(experimentId),
    enabled: enabled && Boolean(experimentId),
    refetchOnWindowFocus: false,
    retry: false,
  });

  const save = useMutation<IssueDetectionSettings, Error, IssueDetectionSettings>({
    // A read already in flight was issued before this write, so it answers with the settings as
    // they were and would land *after* the cache update below -- leaving a run to be submitted
    // with the priorities the user just replaced. Cancelling it first is what makes the write
    // the last word.
    onMutate: () => queryClient.cancelQueries({ queryKey }),
    mutationFn: (settings) => saveIssueDetectionSettings(experimentId, settings),
    // The response is what was persisted, so seeding the cache with it avoids a refetch and
    // avoids briefly showing the pre-save value.
    onSuccess: (saved) => queryClient.setQueryData(queryKey, saved),
  });

  return { ...query, save };
};
