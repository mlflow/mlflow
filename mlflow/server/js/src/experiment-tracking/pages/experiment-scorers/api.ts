import { fetchOrFail, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { catchNetworkErrorIfExists } from '../../utils/NetworkUtils';
import type { ScorerConfig } from './types';

/**
 * Type for the registerScorer API response
 * Matches the protobuf definition in mlflow/protos/service.proto (RegisterScorer.Response)
 */
export interface RegisterScorerResponse {
  version: number;
  scorer_id: string;
  experiment_id: string;
  name: string;
  serialized_scorer: string;
  creation_time: number;
}

/**
 * Type for individual scorer entity in listScorers API response
 * Matches the protobuf definition in mlflow/protos/service.proto (Scorer message)
 */
export interface MLflowScorer {
  experiment_id: number;
  scorer_name: string;
  scorer_version: number;
  serialized_scorer: string;
  creation_time: number;
  scorer_id: string;
}

/**
 * Type for the listScorers API response
 * Matches the protobuf definition in mlflow/protos/service.proto (ListScorers.Response)
 */
export interface ListScorersResponse {
  scorers: MLflowScorer[];
}

/**
 * Get scheduled scorers for an experiment
 */
export async function listScheduledScorers(experimentId: string): Promise<ListScorersResponse> {
  const params = new URLSearchParams();
  params.append('experiment_id', experimentId);
  return fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/scorers/list?${params.toString()}`))
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
  return fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/scorers/update'), {
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
  return fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/scorers/create'), {
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
 * Register a single scorer for an experiment
 */
export async function registerScorer(experimentId: string, scorer: ScorerConfig): Promise<RegisterScorerResponse> {
  return fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/scorers/register'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      experiment_id: experimentId,
      name: scorer.name,
      serialized_scorer: scorer.serialized_scorer,
    }),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}

/**
 * Delete scheduled scorers for an experiment
 */
export async function deleteScheduledScorers(experimentId: string, scorerNames?: string[]) {
  const body: any = {
    experiment_id: experimentId,
  };

  // Add scorer name if provided to delete a specific scorer
  if (scorerNames && scorerNames.length > 0) {
    // Backend expects 'name' parameter for the scorer name
    body.name = scorerNames[0];
  }

  return fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/scorers/delete'), {
    method: 'DELETE',
    body: JSON.stringify(body),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}
