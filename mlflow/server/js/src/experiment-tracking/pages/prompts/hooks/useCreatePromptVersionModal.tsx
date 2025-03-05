import { Alert, FormUI, Modal, RHFControlledComponents, Spacer } from '@databricks/design-system';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import { RegisteredPrompt } from '../types';
import { useCreateRegisteredPromptVersionMutation } from './useCreateRegisteredPromptVersionMutation';

export enum CreatePromptVersionModalMode {
  CreatePrompt = 'CreatePrompt',
  CreatePromptVersion = 'CreatePromptVersion',
}

export const useCreatePromptVersionModal = ({
  mode = CreatePromptVersionModalMode.CreatePromptVersion,
  registeredPrompt,
  onSuccess,
}: {
  mode: CreatePromptVersionModalMode;
  registeredPrompt?: RegisteredPrompt;
  onSuccess?: (result: { promptName: string; promptVersion?: string }) => void | Promise<any>;
}) => {
  const [open, setOpen] = useState(false);
  const intl = useIntl();

  const form = useForm({
    defaultValues: {
      draftName: '',
      draftValue: '',
      commitMessage: '',
    },
  });

  const { mutate: mutateCreateVersion, error, reset: errorsReset } = useCreateRegisteredPromptVersionMutation();

  const modalElement = (
    <Modal
      componentId="TODO"
      visible={open}
      onCancel={() => setOpen(false)}
      title={
        mode === CreatePromptVersionModalMode.CreatePromptVersion ? (
          <FormattedMessage defaultMessage="Create prompt version" description="TODO" />
        ) : (
          <FormattedMessage defaultMessage="Create prompt" description="TODO" />
        )
      }
      okText={<FormattedMessage defaultMessage="Create" description="TODO" />}
      onOk={form.handleSubmit(async (values) => {
        const promptName =
          mode === CreatePromptVersionModalMode.CreatePromptVersion && registeredPrompt?.name
            ? registeredPrompt?.name
            : values.draftName;
        mutateCreateVersion(
          {
            createPromptEntity: mode === CreatePromptVersionModalMode.CreatePrompt,
            content: values.draftValue,
            promptName,
          },
          {
            onSuccess: (data) => {
              const promptVersion = data?.version;
              onSuccess?.({ promptName, promptVersion });
              setOpen(false);
            },
          },
        );
      })}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="TODO" />}
      size="wide"
    >
      {error?.message && (
        <>
          <Alert componentId="TODO" closable={false} message={error.message} type="error" />
          <Spacer />
        </>
      )}
      {mode === CreatePromptVersionModalMode.CreatePrompt && (
        <>
          <FormUI.Label>Name:</FormUI.Label>
          <RHFControlledComponents.Input
            control={form.control}
            componentId="TODO"
            name="draftName"
            rules={{
              required: {
                value: true,
                message: intl.formatMessage({ defaultMessage: 'Name is required', description: 'TODO' }),
              },
            }}
            placeholder={intl.formatMessage({ defaultMessage: 'Provide an unique prompt name', description: 'TODO' })}
            validationState={form.formState.errors.draftName ? 'error' : undefined}
          />
          {form.formState.errors.draftName && (
            <FormUI.Message type="error" message={form.formState.errors.draftName.message} />
          )}
          <Spacer />
        </>
      )}
      <FormUI.Label>Prompt:</FormUI.Label>
      <RHFControlledComponents.TextArea
        control={form.control}
        componentId="TODO"
        name="draftValue"
        autoSize={{ minRows: 3, maxRows: 10 }}
        rules={{
          required: {
            value: true,
            message: intl.formatMessage({ defaultMessage: 'Prompt content is required', description: 'TODO' }),
          },
        }}
        placeholder={intl.formatMessage({ defaultMessage: 'Type prompt content here', description: 'TODO' })}
        validationState={form.formState.errors.draftName ? 'error' : undefined}
      />
      {form.formState.errors.draftValue && (
        <FormUI.Message type="error" message={form.formState.errors.draftValue.message} />
      )}
      <Spacer />
      <FormUI.Label>Commit message:</FormUI.Label>
      <RHFControlledComponents.Input control={form.control} componentId="TODO" name="commitMessage" />
    </Modal>
  );

  const openModal = () => {
    errorsReset();
    form.reset();
    setOpen(true);
  };

  return { CreatePromptModal: modalElement, openModal };
};
