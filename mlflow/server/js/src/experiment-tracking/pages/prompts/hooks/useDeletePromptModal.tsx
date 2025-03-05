import { Modal } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { RegisteredPrompt } from '../types';
import { useDeleteRegisteredPromptMutation } from './useDeleteRegisteredPromptMutation';

export const useDeletePromptModal = ({
  registeredPrompt,
  onSuccess,
}: {
  registeredPrompt?: RegisteredPrompt;
  onSuccess?: () => void | Promise<any>;
}) => {
  const [open, setOpen] = useState(false);

  const { mutate } = useDeleteRegisteredPromptMutation();

  const modalElement = (
    <Modal
      componentId="TODO"
      visible={open}
      onCancel={() => setOpen(false)}
      title={<FormattedMessage defaultMessage="Delete prompt" description="TODO" />}
      okText={<FormattedMessage defaultMessage="Delete" description="TODO" />}
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
      cancelText={<FormattedMessage defaultMessage="Cancel" description="TODO" />}
    >
      <FormattedMessage defaultMessage="Are you sure you want to delete the prompt?" description="TODO" />
    </Modal>
  );

  const openModal = () => setOpen(true);

  return { DeletePromptModal: modalElement, openModal };
};
