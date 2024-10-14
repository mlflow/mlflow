import { DatasetSourceTypes, type DatasetSummary, type RunDatasetWithTags } from '../types';

export const datasetSummariesEqual = (summary1: DatasetSummary, summary2: DatasetSummary) =>
  summary1.digest === summary2.digest && summary1.name === summary2.name && summary1.context === summary2.context;

export const getDatasetSourceUrl = (datasetWithTags: RunDatasetWithTags) => {
  const { dataset } = datasetWithTags;
  const sourceType = dataset.sourceType;
  try {
    if (sourceType === DatasetSourceTypes.HTTP) {
      const { url } = JSON.parse(dataset.source);
      return url;
    }
    if (sourceType === DatasetSourceTypes.S3) {
      const { uri } = JSON.parse(dataset.source);
      return uri;
    }
    if (sourceType === DatasetSourceTypes.HUGGING_FACE) {
      const { path } = JSON.parse(dataset.source);
      return `https://huggingface.co/datasets/${path}`;
    }
  } catch {
    return null;
  }
  return null;
};
