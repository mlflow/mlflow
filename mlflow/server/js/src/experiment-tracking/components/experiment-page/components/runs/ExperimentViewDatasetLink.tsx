import { Button, CopyIcon, TableIcon } from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { DatasetSourceTypes } from '../../../../types';
import { FormattedMessage } from 'react-intl';
import { getDatasetSourceUrl, getEvaluationDatasetId } from '../../../../utils/DatasetUtils';
import { CopyButton } from '../../../../../shared/building_blocks/CopyButton';
import { useNavigate } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';

export interface DatasetLinkProps {
  datasetWithTags: RunDatasetWithTags;
  experimentId?: string;
}

export function ExperimentViewDatasetLink({ datasetWithTags, experimentId }: DatasetLinkProps): JSX.Element | null {
  const { dataset } = datasetWithTags;
  const navigate = useNavigate();

  if (dataset.sourceType === DatasetSourceTypes.EVALUATION_DATASET) {
    const datasetId = getEvaluationDatasetId(datasetWithTags);
    if (datasetId && experimentId) {
      return (
        <Button
          componentId="mlflow.experiment.dataset_drawer.open_evaluation_dataset"
          icon={<TableIcon />}
          type="primary"
          onClick={() => navigate(Routes.getExperimentPageDatasetDetailRoute(experimentId, datasetId))}
        >
          <FormattedMessage
            defaultMessage="Open dataset"
            description="Text for the button that opens the evaluation dataset detail page"
          />
        </Button>
      );
    }
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

  if (
    dataset.sourceType === DatasetSourceTypes.HTTP ||
    dataset.sourceType === DatasetSourceTypes.EXTERNAL ||
    dataset.sourceType === DatasetSourceTypes.HUGGING_FACE
  ) {
    const url = getDatasetSourceUrl(datasetWithTags);
    if (url) {
      return (
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_1"
          icon={<TableIcon />}
          type="primary"
          onClick={() => {
            const newWindow = window.open(url, '_blank', 'noopener,noreferrer');
            if (newWindow) {
              newWindow.opener = null;
            }
          }}
        >
          <FormattedMessage
            defaultMessage="Open dataset"
            description="Text for the button that opens the dataset source URL in a new tab"
          />
        </Button>
      );
    }
  }

  return null;
}
