import { Alert, FormUI, Modal, RHFControlledComponents, Spacer } from '@databricks/design-system';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import type { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { useCreateRegisteredPromptMutation } from './useCreateRegisteredPromptMutation';
import { getPromptContentTagValue } from '../utils';
import { CollapsibleSection } from '@mlflow/mlflow/src/common/components/CollapsibleSection';
import { EditableTagsTableView } from '@mlflow/mlflow/src/common/components/EditableTagsTableView';

export enum CreatePromptModalMode {
  CreatePrompt = 'CreatePrompt',
  CreatePromptVersion = 'CreatePromptVersion',
}

export const useCreatePromptModal = ({
  mode = CreatePromptModalMode.CreatePromptVersion,
  registeredPrompt,
  latestVersion,
  onSuccess,
}: {
  mode: CreatePromptModalMode;
  registeredPrompt?: RegisteredPrompt;
  latestVersion?: RegisteredPromptVersion;
  onSuccess?: (result: { promptName: string; promptVersion?: string }) => void | Promise<any>;
}) => {
  const [open, setOpen] = useState(false);
  const intl = useIntl();

  const form = useForm({
    defaultValues: {
      draftName: '',
      draftValue: '',
      commitMessage: '',
      tags: [] as { key: string; value: string }[],
    },
  });

  const isCreatingNewPrompt = mode === CreatePromptModalMode.CreatePrompt;
  const isCreatingPromptVersion = mode === CreatePromptModalMode.CreatePromptVersion;

  const { mutate: mutateCreateVersion, error, reset: errorsReset, isLoading } = useCreateRegisteredPromptMutation();

  const modalElement = (
    <Modal
      componentId="mlflow.prompts.create.modal"
      visible={open}
      onCancel={() => setOpen(false)}
      title={
        isCreatingPromptVersion ? (
          <FormattedMessage
            defaultMessage="Create prompt version"
            description="A header for the create prompt version modal in the prompt management UI"
          />
        ) : (
          <FormattedMessage
            defaultMessage="Create prompt"
            description="A header for the create prompt modal in the prompt management UI"
          />
        )
      }
      okText={
        <FormattedMessage
          defaultMessage="Create"
          description="A label for the confirm button in the create prompt modal in the prompt management UI"
        />
      }
      okButtonProps={{ loading: isLoading }}
      onOk={form.handleSubmit(async (values) => {
        const promptName =
          isCreatingPromptVersion && registeredPrompt?.name ? registeredPrompt?.name : values.draftName;
        mutateCreateVersion(
          {
            createPromptEntity: isCreatingNewPrompt,
            content: values.draftValue,
            commitMessage: values.commitMessage,
            promptName,
            tags: values.tags,
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
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="A label for the cancel button in the prompt creation modal in the prompt management UI"
        />
      }
      size="wide"
    >
      {error?.message && (
        <>
          <Alert componentId="mlflow.prompts.create.error" closable={false} message={error.message} type="error" />
          <Spacer />
        </>
      )}
      {isCreatingNewPrompt && (
        <>
          <FormUI.Label htmlFor="mlflow.prompts.create.name">Name:</FormUI.Label>
          <RHFControlledComponents.Input
            control={form.control}
            id="mlflow.prompts.create.name"
            componentId="mlflow.prompts.create.name"
            name="draftName"
            rules={{
              required: {
                value: true,
                message: intl.formatMessage({
                  defaultMessage: 'Name is required',
                  description: 'A validation state for the prompt name in the prompt creation modal',
                }),
              },
              pattern: {
                value: /^[a-zA-Z0-9_\-.]+$/,
                message: intl.formatMessage({
                  defaultMessage: 'Only alphanumeric characters, underscores, hyphens, and dots are allowed',
                  description: 'A validation state for the prompt name format in the prompt creation modal',
                }),
              },
            }}
            placeholder={intl.formatMessage({
              defaultMessage: 'Provide an unique prompt name',
              description: 'A placeholder for the prompt name in the prompt creation modal',
            })}
            validationState={form.formState.errors.draftName ? 'error' : undefined}
          />
          {form.formState.errors.draftName && (
            <FormUI.Message type="error" message={form.formState.errors.draftName.message} />
          )}
          <Spacer />
        </>
      )}
      <FormUI.Label htmlFor="mlflow.prompts.create.content">Prompt:</FormUI.Label>
      <RHFControlledComponents.TextArea
        control={form.control}
        id="mlflow.prompts.create.content"
        componentId="mlflow.prompts.create.content"
        name="draftValue"
        autoSize={{ minRows: 3, maxRows: 10 }}
        rules={{
          required: {
            value: true,
            message: intl.formatMessage({
              defaultMessage: 'Prompt content is required',
              description: 'A validation state for the prompt content in the prompt creation modal',
            }),
          },
        }}
        placeholder={intl.formatMessage({
          defaultMessage: "Type prompt content here. Wrap variables with double curly brace e.g. '{{' name '}}'.",
          description: 'A placeholder for the prompt content in the prompt creation modal',
        })}
        validationState={form.formState.errors.draftValue ? 'error' : undefined}
      />
      {form.formState.errors.draftValue && (
        <FormUI.Message type="error" message={form.formState.errors.draftValue.message} />
      )}
      <Spacer />
      <FormUI.Label htmlFor="mlflow.prompts.create.commit_message">Commit message (optional):</FormUI.Label>
      <RHFControlledComponents.Input
        control={form.control}
        id="mlflow.prompts.create.commit_message"
        componentId="mlflow.prompts.create.commit_message"
        name="commitMessage"
      />
    </Modal>
  );

  const openModal = () => {
    errorsReset();
    if (mode === CreatePromptModalMode.CreatePromptVersion && latestVersion) {
      form.reset({
        commitMessage: '',
        draftName: '',
        draftValue: getPromptContentTagValue(latestVersion) ?? '',
        tags: [],
      });
    }
    setOpen(true);
  };

  return { CreatePromptModal: modalElement, openModal };
};
