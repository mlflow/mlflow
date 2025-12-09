/**
 * Hook for editing model configuration of an existing prompt version.
 *
 * Returns a modal component and a function to open it. When opened for a prompt version:
 * - Pre-fills the form with current model config values (if any exist)
 * - Validates user input before saving
 * - Updates the "mlflow.prompt.modelConfig" tag via API on save
 *
 * Usage:
 *   const { EditModelConfigModal, openEditModelConfigModal } = useEditModelConfigModal({ onSuccess: refetch });
 *   // ...
 *   <Button onClick={() => openEditModelConfigModal(promptVersion)}>Edit</Button>
 *   {EditModelConfigModal}
 */
import { useState, useCallback } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { Modal, Button, Spacer, Alert } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { RegisteredPromptsApi } from '../api';
import { ModelConfigForm } from '../components/ModelConfigForm';
import {
  formDataToModelConfig,
  getModelConfigFromTags,
  MLFLOW_PROMPT_MODEL_CONFIG,
  modelConfigToFormData,
  validateModelConfig,
} from '../utils';
import type { PromptModelConfigFormData, RegisteredPromptVersion } from '../types';

export const useEditModelConfigModal = ({ onSuccess }: { onSuccess?: () => void }) => {
  const [open, setOpen] = useState(false);
  const [editingVersion, setEditingVersion] = useState<RegisteredPromptVersion | null>(null);
  const intl = useIntl();

  const form = useForm<{ modelConfig: PromptModelConfigFormData }>({
    defaultValues: { modelConfig: {} },
  });

  const updateMutation = useMutation({
    mutationFn: async ({
      promptName,
      promptVersion,
      modelConfigJson,
    }: {
      promptName: string;
      promptVersion: string;
      modelConfigJson: string;
    }) => {
      return RegisteredPromptsApi.setRegisteredPromptVersionTag(
        promptName,
        promptVersion,
        MLFLOW_PROMPT_MODEL_CONFIG,
        modelConfigJson,
      );
    },
  });

  const openModal = useCallback(
    (version: RegisteredPromptVersion) => {
      updateMutation.reset();
      setEditingVersion(version);
      const currentConfig = getModelConfigFromTags(version.tags);
      form.reset({ modelConfig: modelConfigToFormData(currentConfig) });
      setOpen(true);
    },
    [form, updateMutation],
  );

  const handleSave = form.handleSubmit(async (values) => {
    if (!editingVersion) return;

    // Validate
    const errors = validateModelConfig(values.modelConfig);
    if (Object.keys(errors).length > 0) {
      Object.entries(errors).forEach(([field, message]) => {
        form.setError(`modelConfig.${field}` as any, { type: 'validation', message });
      });
      return;
    }

    // Convert to backend format
    const modelConfig = formDataToModelConfig(values.modelConfig);
    const modelConfigJson = modelConfig ? JSON.stringify(modelConfig) : '{}';

    updateMutation.mutate(
      {
        promptName: editingVersion.name,
        promptVersion: editingVersion.version,
        modelConfigJson,
      },
      {
        onSuccess: () => {
          setOpen(false);
          onSuccess?.();
        },
      },
    );
  });

  const EditModelConfigModal = (
    <FormProvider {...form}>
      <Modal
        componentId="mlflow.prompts.edit_model_config.modal"
        visible={open}
        onCancel={() => {
          setOpen(false);
          form.clearErrors();
        }}
        title={
          <FormattedMessage
            defaultMessage="Edit Model Configuration"
            description="Title for the edit model config modal"
          />
        }
        okText={<FormattedMessage defaultMessage="Save" description="Save button for the edit model config modal" />}
        okButtonProps={{ loading: updateMutation.isLoading }}
        onOk={handleSave}
        cancelText={
          <FormattedMessage defaultMessage="Cancel" description="Cancel button for the edit model config modal" />
        }
        size="normal"
      >
        {updateMutation.error && (
          <>
            <Alert
              componentId="mlflow.prompts.edit_model_config.error"
              closable={false}
              message={(updateMutation.error as Error).message}
              type="error"
            />
            <Spacer />
          </>
        )}
        <ModelConfigForm namePrefix="modelConfig." />
      </Modal>
    </FormProvider>
  );

  return { EditModelConfigModal, openEditModelConfigModal: openModal };
};
