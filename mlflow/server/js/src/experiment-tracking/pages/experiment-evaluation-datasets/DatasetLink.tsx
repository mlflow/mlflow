import { useMemo } from 'react';
import { Link, useLocation, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { SELECTED_DATASET_ID_QUERY_PARAM_KEY } from './hooks/useSelectedDatasetBySearchParam';

export const DatasetLink = ({
  dataset,
  children,
  className,
}: {
  dataset: {
    digest: string;
    name: string;
    profile: string;
    schema: string;
    source: string;
    sourceType: string;
  };
  children: React.ReactElement;
  className?: string;
}) => {
  const { experimentId } = useParams();
  const parsedSource = useMemo<{ table_name?: string; dataset_id?: string } | undefined>(
    () => parseJSONSafe(dataset.source),
    [dataset.source],
  );

  const { search, hash } = useLocation();

  // If the dataset ID is present, render a link to the dataset page
  if (parsedSource?.dataset_id && experimentId) {
    const pathname = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets);
    const searchParams = new URLSearchParams(search);
    searchParams.set(SELECTED_DATASET_ID_QUERY_PARAM_KEY, parsedSource.dataset_id);

    return (
      <Link
        to={{
          pathname,
          search: searchParams.toString(),
          hash,
        }}
        className={className}
      >
        {children}
      </Link>
    );
  }

  // If no link can be rendered, render the children without a link
  return children;
};
