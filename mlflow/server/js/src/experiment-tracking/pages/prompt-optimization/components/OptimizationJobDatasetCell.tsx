import type React from 'react';
import type { ColumnDef } from '@tanstack/react-table';
import { Link, useParams } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { ExperimentPageTabName } from '../../../constants';
import type { PromptOptimizationJob } from '../types';
import { SELECTED_DATASET_ID_QUERY_PARAM_KEY } from '../../experiment-evaluation-datasets/hooks/useSelectedDatasetBySearchParam';
import type { OptimizationJobsTableMetadata } from './OptimizationJobsListTable';

export const OptimizationJobDatasetCell: ColumnDef<PromptOptimizationJob>['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { experimentId } = useParams<{ experimentId: string }>();
  const { getDatasetName } = (meta || {}) as OptimizationJobsTableMetadata;
  const datasetId = original.config?.dataset_id;

  if (!datasetId) {
    return <span>-</span>;
  }

  // Try to get the dataset name from the lookup, fall back to ID
  const datasetName = getDatasetName?.(datasetId) ?? datasetId;

  if (!experimentId) {
    return <span>{datasetName}</span>;
  }

  // Build the link to the dataset page with the dataset selected
  const pathname = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets);
  const searchParams = new URLSearchParams();
  searchParams.set(SELECTED_DATASET_ID_QUERY_PARAM_KEY, datasetId);

  // Stop propagation to prevent row click when clicking the link
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  return (
    <Link
      to={{
        pathname,
        search: searchParams.toString(),
      }}
      onClick={handleClick}
    >
      {datasetName}
    </Link>
  );
};
