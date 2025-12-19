import {
  Alert,
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  FormUI,
  Modal,
  RHFControlledComponents,
  Spacer,
  SegmentedControlGroup,
  SegmentedControlButton,
} from '@databricks/design-system';
import { useState } from 'react';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChatPromptMessage, PromptModelConfigFormData, RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { useCreateRegisteredPromptMutation } from './useCreateRegisteredPromptMutation';
import {
  formDataToModelConfig,
  getChatPromptMessagesFromValue,
  getPromptContentTagValue,
  isChatPrompt,
  PROMPT_EXPERIMENT_IDS_TAG_KEY,
  PROMPT_MODEL_CONFIG_TAG_KEY,
  PROMPT_TYPE_CHAT,
  PROMPT_TYPE_TEXT,
  validateModelConfig,
} from '../utils';
import { ChatMessageCreator } from '../components/ChatMessageCreator';
import { ModelConfigForm } from '../components/ModelConfigForm';

export enum CreatePromptModalMode {
  CreatePrompt = 'CreatePrompt',
  CreatePromptVersion = 'CreatePromptVersion',
}

export const useCreatePromptModal = ({
  mode = CreatePromptModalMode.CreatePromptVersion,
  registeredPrompt,
  latestVersion,
  experimentId,
  onSuccess,
}: {
  mode: CreatePromptModalMode;
  registeredPrompt?: RegisteredPrompt;
  latestVersion?: RegisteredPromptVersion;
  experimentId?: string;
  onSuccess?: (result: { promptName: string; promptVersion?: string }) => void | Promise<any>;
}) => {
  const [open, setOpen] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const intl = useIntl();

  const form = useForm<{
    draftName: string;
    draftValue: string;
    chatMessages: ChatPromptMessage[];
    commitMessage: string;
    tags: { key: string; value: string }[];
    promptType: typeof PROMPT_TYPE_CHAT | typeof PROMPT_TYPE_TEXT;
    modelConfig: PromptModelConfigFormData;
  }>({
    defaultValues: {
      draftName: '',
      draftValue: '',
      chatMessages: [{ role: 'user', content: '' }],
      commitMessage: '',
      tags: [],
      promptType: PROMPT_TYPE_TEXT,
      modelConfig: {},
    },
  });

  const isCreatingNewPrompt = mode === CreatePromptModalMode.CreatePrompt;
  const isCreatingPromptVersion = mode === CreatePromptModalMode.CreatePromptVersion;

  const { mutate: mutateCreateVersion, error, reset: errorsReset, isLoading } = useCreateRegisteredPromptMutation();

  const modalElement = (
    <FormProvider {...form}>
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

          if (values.promptType === PROMPT_TYPE_CHAT) {
            const hasEmptyMessage = values.chatMessages.some((m) => !m.content || !m.content.trim());
            if (hasEmptyMessage) {
              form.setError('chatMessages', {
                type: 'required',
                message: intl.formatMessage({
                  defaultMessage: 'Prompt content is required',
                  description: 'A validation state for the chat prompt content in the prompt creation modal',
                }),
              });
              return;
            }
          }

          // Validate model config if any field is provided
          const modelConfigErrors = validateModelConfig(values.modelConfig);
          if (Object.keys(modelConfigErrors).length > 0) {
            Object.entries(modelConfigErrors).forEach(([field, message]) => {
              form.setError(`modelConfig.${field}` as any, { type: 'validation', message });
            });
            return;
          }

          const chatMessages = values.chatMessages.map((message) => ({
            ...message,
            content: message.content.trim(),
          }));

          // Prepare experiment ID tag which has `,${experimentId},` format
          const promptTags = experimentId ? [{ key: PROMPT_EXPERIMENT_IDS_TAG_KEY, value: `,${experimentId},` }] : [];

          // Convert model config form data to backend format
          const modelConfig = formDataToModelConfig(values.modelConfig);
          const modelConfigTags = modelConfig
            ? [{ key: PROMPT_MODEL_CONFIG_TAG_KEY, value: JSON.stringify(modelConfig) }]
            : [];

          mutateCreateVersion(
            {
              createPromptEntity: isCreatingNewPrompt,
              content: values.promptType === PROMPT_TYPE_CHAT ? JSON.stringify(chatMessages) : values.draftValue,
              commitMessage: values.commitMessage,
              promptName,
              tags: [...values.tags, ...modelConfigTags],
              promptTags,
              promptType: values.promptType,
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
        <FormUI.Label>
          <FormattedMessage
            defaultMessage="Prompt type:"
            description="A label for selecting prompt type in the prompt creation modal"
          />
        </FormUI.Label>
        <Controller
          control={form.control}
          name="promptType"
          render={({ field }) => (
            <SegmentedControlGroup
              name="promptType"
              componentId="promptType"
              value={field.value}
              onChange={field.onChange}
            >
              <SegmentedControlButton value={PROMPT_TYPE_TEXT}>
                <FormattedMessage
                  defaultMessage="Text"
                  description="Label for text prompt type in the prompt creation modal"
                />
              </SegmentedControlButton>
              <SegmentedControlButton value={PROMPT_TYPE_CHAT}>
                <FormattedMessage
                  defaultMessage="Chat"
                  description="Label for chat prompt type in the prompt creation modal"
                />
              </SegmentedControlButton>
            </SegmentedControlGroup>
          )}
        />
        <Spacer />
        <FormUI.Label htmlFor="mlflow.prompts.create.content">Prompt:</FormUI.Label>
        {form.watch('promptType') === PROMPT_TYPE_CHAT ? (
          <ChatMessageCreator name="chatMessages" />
        ) : (
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
        )}
        {form.watch('promptType') === PROMPT_TYPE_TEXT && form.formState.errors.draftValue && (
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
        <Spacer />
        {/* Advanced Settings Section */}
        <Button
          componentId="mlflow.prompts.create.toggle_advanced_settings"
          type="link"
          onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
          icon={showAdvancedSettings ? <ChevronDownIcon /> : <ChevronRightIcon />}
          css={{ padding: 0 }}
        >
          <FormattedMessage
            defaultMessage="Advanced settings (optional)"
            description="Toggle button for advanced settings in prompt creation modal"
          />
        </Button>
        {showAdvancedSettings && (
          <>
            <Spacer size="sm" />
            <ModelConfigForm />
          </>
        )}
      </Modal>
    </FormProvider>
  );

  const openModal = () => {
    errorsReset();
    const tagValue =
      mode === CreatePromptModalMode.CreatePromptVersion && latestVersion
        ? getPromptContentTagValue(latestVersion) ?? ''
        : '';
    const promptType = isChatPrompt(latestVersion) ? PROMPT_TYPE_CHAT : PROMPT_TYPE_TEXT;
    const parsedMessages = getChatPromptMessagesFromValue(tagValue);

    // Check if latest version has model config to auto-expand advanced settings
    const hasModelConfig =
      mode === CreatePromptModalMode.CreatePromptVersion &&
      latestVersion?.tags?.some((tag) => tag.key === PROMPT_MODEL_CONFIG_TAG_KEY && tag.value);

    form.reset({
      commitMessage: '',
      draftName: '',
      draftValue: parsedMessages ? '' : tagValue,
      chatMessages: parsedMessages
        ? parsedMessages.map((message) => ({ ...message }))
        : [{ role: 'user', content: '' }],
      tags: [],
      promptType,
      modelConfig: {},
    });
    setShowAdvancedSettings(!!hasModelConfig);
    setOpen(true);
  };

  return { CreatePromptModal: modalElement, openModal };
};
