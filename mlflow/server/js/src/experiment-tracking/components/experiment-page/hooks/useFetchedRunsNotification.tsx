import type { NotificationInstance } from '@databricks/design-system';
import { useCallback } from 'react';
import { useIntl } from 'react-intl';
import type { RunEntity, RunInfoEntity } from '../../../types';
import { EXPERIMENT_PARENT_ID_TAG } from '../utils/experimentPage.common-utils';

const FETCHED_RUN_NOTIFICATION_DURATION = 3; // Seconds
const FETCHED_RUN_NOTIFICATION_KEY = 'FETCHED_RUN_NOTIFICATION_KEY';

const countFetchedRuns = (fetchedRuns: RunEntity[], existingRunInfos: RunInfoEntity[] = []) => {
  // Extract only runs that are not loaded yet
  const newRuns = fetchedRuns.filter((r) => !existingRunInfos.some((x) => x.runUuid === r.info.runUuid));

  // Next, extract runs containing non-empty "parentRunId" tag
  const runsWithParent = newRuns.filter((run: any) => {
    const runTagsList = run?.data?.tags;
    return (
      Array.isArray(runTagsList) &&
      runTagsList.some((tag) => tag.key === EXPERIMENT_PARENT_ID_TAG && Boolean(tag.value))
    );
  });

  // Return counts of both all runs and those with parent
  return {
    allRuns: newRuns.length,
    childRuns: runsWithParent.length,
  };
};

export const useFetchedRunsNotification = (notification: NotificationInstance) => {
  const { formatMessage } = useIntl();

  // Creates the localized message based on the returned run count
  const getMessage = useCallback(
    (allRuns: number, childRuns: number) => {
      // Returned when only child runs are loaded
      if (allRuns === childRuns) {
        return formatMessage(
          {
            defaultMessage: 'Loaded {childRuns} child {childRuns, plural, =1 {run} other {runs}}',
            description: 'Experiment page > loaded more runs notification > loaded only child runs',
          },
          { childRuns: childRuns },
        );
      }

      // Returned when we fetch both regular (parent) and child runs
      return formatMessage(
        {
          defaultMessage:
            // eslint-disable-next-line formatjs/no-multiple-plurals
            'Loaded {allRuns} {allRuns, plural, =1 {run} other {runs}}, including {childRuns} child {childRuns, plural, =1 {run} other {runs}}',
          description: 'Experiment page > loaded more runs notification > loaded both parent and child runs',
        },
        { allRuns, childRuns: childRuns },
      );
    },
    [formatMessage],
  );

  return useCallback(
    (fetchedRuns: RunEntity[], existingRunInfos: RunInfoEntity[]) => {
      if (Array.isArray(fetchedRuns)) {
        // Get counted runs
        const { allRuns, childRuns } = countFetchedRuns(fetchedRuns, existingRunInfos);

        // Display notification only if there are any new child runs
        if (childRuns < 1) {
          return;
        }

        // If there is a similar notification visible already, close it first
        // to avoid confusion due to multiple displayed notification elements
        notification.close(FETCHED_RUN_NOTIFICATION_KEY);

        // Display the notification
        notification.info({
          message: getMessage(allRuns, childRuns),
          duration: FETCHED_RUN_NOTIFICATION_DURATION,
          placement: 'bottomRight',
          key: FETCHED_RUN_NOTIFICATION_KEY,
        });
      }
    },
    [notification, getMessage],
  );
};
