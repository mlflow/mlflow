import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { Modal } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import type { RegisteredPrompt } from '../types';
import { RegisteredPromptsApi } from '../api';

export const useDeletePromptModal = ({
  registeredPrompt,
  onSuccess,
}: {
  registeredPrompt?: RegisteredPrompt;
  onSuccess?: () => void | Promise<any>;
}) => {
  const [open, setOpen] = useState(false);

  const { mutate } = useMutation<
    unknown,
    Error,
    {
      promptName: string;
    }
  >({
    mutationFn: async ({ promptName }) => {
      await RegisteredPromptsApi.deleteRegisteredPrompt(promptName);
    },
  });

  const modalElement = (
    <Modal
      componentId="mlflow.prompts.delete_modal"
      visible={open}
      onCancel={() => setOpen(false)}
      title={<FormattedMessage defaultMessage="Delete prompt" description="A header for the delete prompt modal" />}
      okText={
        <FormattedMessage
          defaultMessage="Delete"
          description="A label for the confirm button in the delete prompt modal"
        />
      }
      okButtonProps={{ danger: true }}
      onOk={async () => {
        if (!registeredPrompt?.name) {
          setOpen(false);
          return;
        }
        mutate(
          {
            promptName: registeredPrompt.name,
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
          description="A label for the cancel button in the delete prompt modal"
        />
      }
    >
      <FormattedMessage
        defaultMessage="Are you sure you want to delete the prompt?"
        description="A content for the delete prompt confirmation modal"
      />
    </Modal>
  );

  const openModal = () => setOpen(true);

  return { DeletePromptModal: modalElement, openModal };
};
