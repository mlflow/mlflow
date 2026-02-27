import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { Modal } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import type { RegisteredPromptVersion } from '../types';
import { RegisteredPromptsApi } from '../api';

export const useDeletePromptVersionModal = ({
  promptVersion,
  onSuccess,
}: {
  promptVersion?: RegisteredPromptVersion;
  onSuccess?: () => void | Promise<any>;
}) => {
  const [open, setOpen] = useState(false);

  const { mutate } = useMutation<
    unknown,
    Error,
    {
      promptName: string;
      version: string;
    }
  >({
    mutationFn: async ({ promptName, version }) => {
      await RegisteredPromptsApi.deleteRegisteredPromptVersion(promptName, version);
    },
  });

  const modalElement = (
    <Modal
      componentId="mlflow.prompts.delete_version_modal"
      visible={open}
      onCancel={() => setOpen(false)}
      title={
        <FormattedMessage
          defaultMessage="Delete prompt version"
          description="A header for the delete prompt version modal"
        />
      }
      okText={
        <FormattedMessage
          defaultMessage="Delete"
          description="A label for the confirm button in the delete prompt version modal"
        />
      }
      okButtonProps={{ danger: true }}
      onOk={async () => {
        if (!promptVersion?.name) {
          setOpen(false);
          return;
        }
        mutate(
          {
            promptName: promptVersion.name,
            version: promptVersion.version,
          },
          {
            onSuccess: () => {
              onSuccess?.();
              setOpen(false);
            },
          },
        );
        setOpen(false);
      }}
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="A label for the cancel button in the delete prompt version modal"
        />
      }
    >
      <FormattedMessage
        defaultMessage="Are you sure you want to delete the prompt version?"
        description="A content for the delete prompt version confirmation modal"
      />
    </Modal>
  );

  const openModal = () => setOpen(true);

  return { DeletePromptModal: modalElement, openModal };
};
