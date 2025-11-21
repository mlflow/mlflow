import { fetchOrFail } from '../../../common/utils/FetchUtils';
import { catchNetworkErrorIfExists } from '../../utils/NetworkUtils';
import type { ScorerConfig } from './types';

/**
 * Get scheduled scorers for an experiment
 */
export async function listScheduledScorers(experimentId: string) {
  const params = new URLSearchParams();
  params.append('experiment_id', experimentId);
  return fetchOrFail(`/ajax-api/3.0/mlflow/scorers/list?${params.toString()}`)
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}

/**
 * Update scheduled scorers for an experiment
 */
export async function updateScheduledScorers(
  experimentId: string,
  scheduledScorers: {
    scorers: ScorerConfig[];
  },
  updateMask: string = 'scheduled_scorers.scorers',
) {
  return fetchOrFail(`/ajax-api/3.0/mlflow/scorers/update`, {
    method: 'PATCH',
    body: JSON.stringify({
      experiment_id: experimentId,
      scheduled_scorers: scheduledScorers,
      update_mask: updateMask,
    }),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}

/**
 * Create scheduled scorers for an experiment
 */
export async function createScheduledScorers(
  experimentId: string,
  scheduledScorers: {
    scorers: ScorerConfig[];
  },
) {
  return fetchOrFail(`/ajax-api/3.0/mlflow/scorers/create`, {
    method: 'POST',
    body: JSON.stringify({
      experiment_id: experimentId,
      scheduled_scorers: scheduledScorers,
    }),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}

/**
 * Delete scheduled scorers for an experiment
 */
export async function deleteScheduledScorers(experimentId: string) {
  const params = new URLSearchParams();
  params.append('experiment_id', experimentId);
  return fetchOrFail(`/ajax-api/3.0/mlflow/scorers/delete?${params.toString()}`, {
    method: 'DELETE',
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}
