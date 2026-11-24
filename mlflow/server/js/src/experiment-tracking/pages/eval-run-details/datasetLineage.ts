import { parseJSONSafe } from '../../../common/utils/TagUtils';

/**
 * The ID of the MLflow evaluation dataset an input dataset was logged from: its source is
 * `{"dataset_id": ...}` (EvaluationDatasetSource in mlflow/data/evaluation_dataset_source.py). Undefined
 * for other sources, such as code or files, which have no dataset page to link to.
 */
export const getEvaluationDatasetId = (source: string | null | undefined): string | undefined => {
  if (!source) {
    return undefined;
  }
  const parsedSource = parseJSONSafe(source);
  return parsedSource && typeof parsedSource === 'object' && typeof parsedSource.dataset_id === 'string'
    ? parsedSource.dataset_id || undefined
    : undefined;
};
