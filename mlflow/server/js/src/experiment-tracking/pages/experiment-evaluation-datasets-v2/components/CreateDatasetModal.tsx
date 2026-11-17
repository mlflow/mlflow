import { Button, DatabaseIcon, Tooltip } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState } from 'react';
import { CreateEvaluationDatasetModal } from '../../experiment-evaluation-datasets/components/CreateEvaluationDatasetModal';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { useCanCreateDataset } from '../hooks/useCanCreateDataset';

/**
 * Trigger + modal for "Create dataset" on the v2 datasets list page. Wraps OSS's existing
 * `CreateEvaluationDatasetModal` with the button surface the v2 toolbar expects. The wrapped
 * modal owns its own success path (cache invalidation via `useCreateEvaluationDatasetMutation`),
 * so the `onSuccess` / `refetch` props from universe call sites are accepted but unused —
 * threading the created dataset back through to the caller is tracked as a followup.
 */
interface CreateDatasetButtonProps {
  experimentId: string;
  onSuccess?: (dataset: Dataset) => void;
  refetch?: () => void;
  buttonText?: React.ReactNode;
  buttonProps?: Record<string, unknown>;
}

export const CreateDatasetButton = ({ experimentId, buttonText, buttonProps = {} }: CreateDatasetButtonProps) => {
  const [open, setOpen] = useState(false);
  const canCreate = useCanCreateDataset(experimentId);
  const createButton = (
    <Button
      componentId="mlflow.eval-datasets-v2.create-dataset.button"
      icon={<DatabaseIcon />}
      type="primary"
      {...buttonProps}
      disabled={!canCreate || Boolean(buttonProps['disabled'])}
      onClick={() => setOpen(true)}
    >
      {buttonText ?? (
        <FormattedMessage
          defaultMessage="Create dataset"
          description="Button label that opens the create-dataset modal on the v2 datasets list"
        />
      )}
    </Button>
  );
  return (
    <>
      {canCreate ? (
        createButton
      ) : (
        <Tooltip
          componentId="mlflow.eval-datasets-v2.create-dataset.permission-tooltip"
          content={
            <FormattedMessage
              defaultMessage="Edit permission on this experiment must be confirmed before you can create a dataset."
              description="Tooltip explaining why dataset creation is disabled"
            />
          }
        >
          <span>{createButton}</span>
        </Tooltip>
      )}
      {open && <CreateEvaluationDatasetModal visible experimentId={experimentId} onCancel={() => setOpen(false)} />}
    </>
  );
};
