import { Button, CopyIcon, TableIcon } from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { DatasetSourceTypes } from '../../../../types';
import { FormattedMessage } from 'react-intl';
import { getDatasetSourceUrl } from '../../../../utils/DatasetUtils';
import { CopyButton } from '../../../../../shared/building_blocks/CopyButton';
import { Link } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import { ExperimentPageTabName } from '../../../../constants';

export interface DatasetLinkProps {
  datasetWithTags: RunDatasetWithTags;
  runTags: Record<string, { key: string; value: string }>;
  experimentId?: string;
}

export function ExperimentViewDatasetLink({ datasetWithTags, experimentId }: DatasetLinkProps): JSX.Element | null {
  const { dataset } = datasetWithTags;

  if (experimentId) {
    const datasetsRoute = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets);
    return (
      <Button
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_1"
        icon={<TableIcon />}
        type="primary"
      >
        <Link
          componentId="mlflow.experiment_tracking.dataset_link.open_dataset_link"
          to={datasetsRoute}
          css={{ color: 'inherit', '&:hover': { color: 'inherit', textDecoration: 'none' } }}
        >
          <FormattedMessage
            defaultMessage="Open dataset"
            description="Text for the button that navigates to the datasets tab in the experiment run dataset drawer"
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
}
