import { Button, DatabaseIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState } from 'react';
import { CreateEvaluationDatasetModal } from '../../experiment-evaluation-datasets/components/CreateEvaluationDatasetModal';
import type { Dataset } from '../hooks/useDatasetsQueries';

interface CreateDatasetButtonProps {
  experimentId: string;
  /**
   * Universe fires this with the newly-created dataset so callers can route into it. OSS
   * doesn't thread the created dataset through `CreateEvaluationDatasetModal` here yet, so
   * the callback is currently never invoked — list pages rely on the mutation hook's cache
   * invalidation to surface the new row. Prop is kept for shape parity with universe.
   */
  onSuccess?: (dataset: Dataset) => void;
  refetch?: () => void;
  buttonText?: React.ReactNode;
  /** Pass-through to the underlying Button — `css` allowed. */
  buttonProps?: Record<string, unknown>;
}

/**
 * Trigger + modal for "Create dataset" on the v2 datasets list page.
 *
 * OSS-only adapter: wraps `CreateEvaluationDatasetModal` (the existing OSS modal) with the
 * button surface the v2 toolbar expects. The modal handles its own success path via
 * `useCreateEvaluationDatasetMutation`, so `onSuccess` / `refetch` callbacks are no-ops in
 * OSS — kept for prop-shape parity with universe.
 */
export const CreateDatasetButton = ({ experimentId, buttonText, buttonProps = {} }: CreateDatasetButtonProps) => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        componentId="mlflow.eval-datasets-v2.create-dataset.button"
        icon={<DatabaseIcon />}
        type="primary"
        onClick={() => setOpen(true)}
        {...buttonProps}
      >
        {buttonText ?? (
          <FormattedMessage
            defaultMessage="Create dataset"
            description="Button label that opens the create-dataset modal on the v2 datasets list"
          />
        )}
      </Button>
      <CreateEvaluationDatasetModal visible={open} experimentId={experimentId} onCancel={() => setOpen(false)} />
    </>
  );
};
