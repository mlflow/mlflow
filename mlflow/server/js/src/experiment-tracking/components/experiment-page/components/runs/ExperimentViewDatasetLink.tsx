import { Button, CopyIcon, TableIcon } from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { DatasetSourceTypes } from '../../../../types';
import { MLFLOW_RUN_DATASET_CONTEXT_TAG } from '../../../../constants';
import { FormattedMessage } from 'react-intl';
import { getDatasetSourceUrl } from '../../../../utils/DatasetUtils';
import { CopyButton } from '../../../../../shared/building_blocks/CopyButton';
import { Link } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import { ExperimentPageTabName } from '../../../../constants';
import { btoaUtf8 } from '../../../../../common/utils/StringUtils';

export interface DatasetLinkProps {
  datasetWithTags: RunDatasetWithTags;
  runTags: Record<string, { key: string; value: string }>;
  experimentId?: string;
}

/**
 * Builds a URL to the experiment Runs tab with the datasetsFilter pre-set
 * to filter by the given dataset (name + digest + context).
 */
const buildFilteredRunsRoute = (experimentId: string, datasetWithTags: RunDatasetWithTags): string => {
  const { dataset, tags } = datasetWithTags;
  const context = tags?.find(({ key }) => key === MLFLOW_RUN_DATASET_CONTEXT_TAG)?.value;
  const filterPayload = [{ name: dataset.name, digest: dataset.digest, context }];
  const encodedFilter = btoaUtf8(JSON.stringify(filterPayload));
  const runsRoute = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Runs);
  return `${runsRoute}?datasetsFilter=${encodeURIComponent(encodedFilter)}`;
};

export const ExperimentViewDatasetLink = ({ datasetWithTags, runTags, experimentId }: DatasetLinkProps) => {
  const { dataset } = datasetWithTags;

  if (experimentId) {
    return (
      <Button
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_1"
        icon={<TableIcon />}
        type="primary"
      >
        <Link
          to={buildFilteredRunsRoute(experimentId, datasetWithTags)}
          css={{ color: 'inherit', '&:hover': { color: 'inherit', textDecoration: 'none' } }}
        >
          <FormattedMessage
            defaultMessage="View runs with dataset"
            description="Text for the button that navigates to runs filtered by this dataset in the experiment run dataset drawer"
          />
        </Link>
      </Button>
    );
  }

  if (dataset.sourceType === DatasetSourceTypes.S3) {
    const url = getDatasetSourceUrl(datasetWithTags);
    if (url) {
      return (
        <CopyButton
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_2"
          icon={<CopyIcon />}
          copyText={url}
        >
          <FormattedMessage
            defaultMessage="Copy S3 URI to clipboard"
            description="Text for the S3 URI copy button in the experiment run dataset drawer"
          />
        </CopyButton>
      );
    }
  }
  return null;
};
