import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { Alert, Modal, Spacer } from '@databricks/design-system';
import { useCallback, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import type { LoggedModelProto } from '../../../types';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const useExperimentLoggedModelDeleteModal = ({
  loggedModel,
  onSuccess,
}: {
  loggedModel?: LoggedModelProto | null;
  onSuccess?: () => void | Promise<any>;
}) => {
  const [open, setOpen] = useState(false);

  const mutation = useMutation<
    unknown,
    Error,
    {
      loggedModelId: string;
    }
  >({
    mutationFn: async ({ loggedModelId }) => {
      await fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/logged-models/${loggedModelId}`), 'DELETE');
    },
  });

  const { mutate, isLoading, reset: resetMutation } = mutation;

  const modalElement = (
    <Modal
      componentId="mlflow.logged_model.details.delete_modal"
      visible={open}
      onCancel={() => setOpen(false)}
      title={
        <FormattedMessage
          defaultMessage="Delete logged model"
          description="A header of the modal used for deleting logged models"
        />
      }
      okText={
        <FormattedMessage
          defaultMessage="Delete"
          description="A confirmation label of the modal used for deleting logged models"
        />
      }
      okButtonProps={{ danger: true, loading: isLoading }}
      onOk={async () => {
        if (!loggedModel?.info?.model_id) {
          setOpen(false);
          return;
        }
        mutate(
          {
            loggedModelId: loggedModel.info.model_id,
          },
          {
            onSuccess: () => {
              onSuccess?.();
              setOpen(false);
            },
          },
        );
      }}
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="A cancel label for the modal used for deleting logged models"
        />
      }
    >
      {mutation.error?.message && (
        <>
          <Alert
            componentId="mlflow.logged_model.details.delete_modal.error"
            closable={false}
            message={mutation.error.message}
            type="error"
          />
          <Spacer />
        </>
      )}
      <FormattedMessage
        defaultMessage="Are you sure you want to delete this logged model?"
        description="A content of the delete logged model confirmation modal"
      />
    </Modal>
  );

  const openModal = useCallback(() => {
    resetMutation();
    setOpen(true);
  }, [resetMutation]);

  return { modalElement, openModal };
};
