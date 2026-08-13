import { DatasetSourceTypes, type DatasetSummary, type RunDatasetWithTags } from '../types';

export function datasetSummariesEqual(summary1: DatasetSummary, summary2: DatasetSummary): boolean {
  return (
    summary1.digest === summary2.digest && summary1.name === summary2.name && summary1.context === summary2.context
  );
}

/**
 * Extracts the evaluation dataset id from a run input dataset. Evaluation datasets
 * (`mlflow.genai.datasets`) store the id inside the `source` JSON rather than as a
 * top-level field, so it must be parsed out to link to the dataset detail page.
 */
export function getEvaluationDatasetId(datasetWithTags: RunDatasetWithTags): string | null {
  const { dataset } = datasetWithTags;
  if (dataset.sourceType !== DatasetSourceTypes.EVALUATION_DATASET) {
    return null;
  }
  try {
    return JSON.parse(dataset.source)?.dataset_id ?? null;
  } catch {
    return null;
  }
}

export function getDatasetSourceUrl(datasetWithTags: RunDatasetWithTags): string | null {
  const { dataset } = datasetWithTags;

  try {
    const parsed = JSON.parse(dataset.source);

    switch (dataset.sourceType) {
      case DatasetSourceTypes.HTTP:
      case DatasetSourceTypes.EXTERNAL:
        return parsed.url ?? null;
      case DatasetSourceTypes.S3:
        return parsed.uri ?? null;
      case DatasetSourceTypes.HUGGING_FACE:
        return parsed.path ? `https://huggingface.co/datasets/${parsed.path}` : null;
      case DatasetSourceTypes.LOCAL:
        return parsed.uri ?? null;
      default:
        return null;
    }
  } catch {
    return null;
  }
}
