import { random } from 'lodash';
import type { LoggedModelProto } from '../types';
import { LoggedModelStatusProtoEnum } from '../types';
import { generateRandomRunName } from './RunNameUtils';
import { rest } from 'msw';

// Generate some demo data
const getLoggedModelsDemoData = (experimentId: string) =>
  new Array(50).fill(0).map<LoggedModelProto>((_, index) => ({
    info: {
      artifact_uri: `dbfs:/databricks/mlflow/${'model-' + (index + 1)}`,
      creation_timestamp_ms: 1728322600000,
      last_updated_timestamp_ms: 1728322600000,
      source_run_id: 'run-id-1',
      creator_id: 'test@test.com',
      experiment_id: experimentId,
      model_id: `m-${index + 1}`,
      model_type: 'Agent',
      name: 'model-' + (index + 1),
      tags: [
        {
          key: 'mlflow.sourceRunName',
          value: generateRandomRunName(),
        },
      ],
      status_message: 'Ready',
      status:
        index % 20 === 7
          ? LoggedModelStatusProtoEnum.LOGGED_MODEL_UPLOAD_FAILED
          : index % 10 === 1
          ? LoggedModelStatusProtoEnum.LOGGED_MODEL_PENDING
          : LoggedModelStatusProtoEnum.LOGGED_MODEL_READY,
      registrations: [],
    },
    data: {
      metrics: new Array(20).fill(0).map((_, index) => {
        // Cycle through 5 run IDs
        const run_index = (index % 5) + 1;
        const run_id = `run-id-${run_index}`;
        // First two runs will have datasets 1 and 2, other will get 11 and 12
        const dataset_digest = String((run_index > 2 ? 10 : 0) + Math.floor(index / 10) + 1);
        const dataset_name = `dataset-${dataset_digest}`;
        return {
          key: 'metric-' + (index + 1),
          value: random(-50, 50, true),
          step: 1,
          timestamp: 1728322600000,
          dataset_digest,
          dataset_name,
          model_id: (index + 1).toString(),
          run_id,
        };
      }),
      params: [
        {
          key: 'top_k',
          value: '0.9',
        },
        {
          key: 'generative_llm',
          value: 'GPT-4',
        },
        {
          key: 'max_tokens',
          value: '2000',
        },
      ],
    },
  }));

export const mockSearchLoggedModels = (
  experimentId = 'test-experiment',
  models = getLoggedModelsDemoData(experimentId),
) =>
  rest.post('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) =>
    res(
      ctx.json({
        models,
      }),
    ),
  );

export const mockGetLoggedModels = (models = getLoggedModelsDemoData('test-experiment')) =>
  rest.get('/ajax-api/2.0/mlflow/logged-models:batchGet', (req, res, ctx) =>
    res(
      ctx.json({
        models,
      }),
    ),
  );
