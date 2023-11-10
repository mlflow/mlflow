import type { DatasetSummary } from '../types';

export const datasetSummariesEqual = (summary1: DatasetSummary, summary2: DatasetSummary) =>
  summary1.digest === summary2.digest &&
  summary1.name === summary2.name &&
  summary1.context === summary2.context;
