import { Button, Spinner, TableIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useState } from 'react';
import { useExperimentLoggedModelOpenDatasetDetails } from './hooks/useExperimentLoggedModelOpenDatasetDetails';
import { useUserActionErrorHandler } from '@databricks/web-shared/metrics';

export const ExperimentLoggedModelDatasetButton = ({
  datasetName,
  datasetDigest,
  runId,
}: {
  datasetName: string;
  datasetDigest: string;
  runId: string | null;
}) => {
  const { theme } = useDesignSystemTheme();
  const [loadingDatasetDetails, setLoadingDatasetDetails] = useState(false);
  const { onDatasetClicked } = useExperimentLoggedModelOpenDatasetDetails();
  const { handleError } = useUserActionErrorHandler();

  const handleDatasetClick = (datasetName: string, datasetDigest: string, runId: string | null) => {
    if (runId) {
      setLoadingDatasetDetails(true);
      onDatasetClicked?.({ datasetName, datasetDigest, runId })
        .catch((error) => {
          handleError(error);
        })
        .finally(() => setLoadingDatasetDetails(false));
    }
  };

  return (
    <Button
      type="link"
      icon={loadingDatasetDetails ? <Spinner size="small" css={{ marginRight: theme.spacing.sm }} /> : <TableIcon />}
      key={[datasetName, datasetDigest].join('.')}
      componentId="mlflow.logged_model.dataset"
      onClick={() => handleDatasetClick(datasetName, datasetDigest, runId)}
    >
      {datasetName} (#{datasetDigest})
    </Button>
  );
};
