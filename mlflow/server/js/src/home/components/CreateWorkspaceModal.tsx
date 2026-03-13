import { useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import { Alert, FormUI, Modal, RHFControlledComponents, Spacer } from '@databricks/design-system';
import { fetchAPI, getAjaxUrl, HTTPMethods } from '../../common/utils/FetchUtils';
import { validateWorkspaceName } from '../../workspaces/utils/WorkspaceUtils';

type CreateWorkspaceFormData = {
  workspaceName: string;
  workspaceDescription?: string;
  workspaceArtifactRoot?: string;
};

export const useCreateWorkspaceModal = ({ onSuccess }: { onSuccess?: (workspaceName: string) => void }) => {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const intl = useIntl();

  const form = useForm<CreateWorkspaceFormData>({
    defaultValues: {
      workspaceName: '',
      workspaceDescription: '',
      workspaceArtifactRoot: '',
    },
  });

  const handleSubmit = async (values: CreateWorkspaceFormData) => {
    setError(null);
    setIsLoading(true);

    const requestBody: { name: string; description?: string; default_artifact_root?: string } = {
      name: values.workspaceName,
    };

    if (values.workspaceDescription) {
      requestBody.description = values.workspaceDescription;
    }

    if (values.workspaceArtifactRoot) {
      requestBody.default_artifact_root = values.workspaceArtifactRoot;
    }

    try {
      await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/workspaces'), {
        method: HTTPMethods.POST,
        body: JSON.stringify(requestBody),
      });

      onSuccess?.(values.workspaceName);
      setOpen(false);
      form.reset();
    } catch (err: any) {
      setError(err?.message || 'Failed to create workspace');
    } finally {
      setIsLoading(false);
    }
  };

  const modalElement = (
    <FormProvider {...form}>
      <Modal
        componentId="mlflow.home.create_workspace_modal"
        visible={open}
        onCancel={() => setOpen(false)}
        title={<FormattedMessage defaultMessage="Create Workspace" description="Title for create workspace modal" />}
        okText={
          <FormattedMessage defaultMessage="Create" description="Confirm button text for create workspace modal" />
        }
        okButtonProps={{ loading: isLoading }}
        onOk={form.handleSubmit(handleSubmit)}
        cancelText={
          <FormattedMessage defaultMessage="Cancel" description="Cancel button text for create workspace modal" />
        }
      >
        <div
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              form.handleSubmit(handleSubmit)();
            }
          }}
        >
          {error && (
            <>
              <Alert
                componentId="mlflow.home.create_workspace_modal.error"
                closable={false}
                message={error}
                type="error"
              />
              <Spacer />
            </>
          )}
          <FormUI.Label htmlFor="mlflow.home.create_workspace_modal.workspace_name">
            <FormattedMessage defaultMessage="Workspace Name" description="Label for workspace name field" />:
          </FormUI.Label>
          <RHFControlledComponents.Input
            control={form.control}
            id="mlflow.home.create_workspace_modal.workspace_name"
            componentId="mlflow.home.create_workspace_modal.workspace_name_input"
            name="workspaceName"
            autoFocus
            rules={{
              required: {
                value: true,
                message: intl.formatMessage({
                  defaultMessage: 'Please input a name for the new workspace.',
                  description: 'Error message for name requirement in create workspace modal',
                }),
              },
              validate: (value) => {
                if (!value) {
                  return true; // Let required rule handle empty values
                }
                const result = validateWorkspaceName(value);
                return result.valid || result.error;
              },
            }}
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter workspace name',
              description: 'Input placeholder for workspace name in create workspace modal',
            })}
            validationState={form.formState.errors.workspaceName ? 'error' : undefined}
          />
          {form.formState.errors.workspaceName && (
            <FormUI.Message type="error" message={form.formState.errors.workspaceName.message} />
          )}
          <Spacer />
          <FormUI.Label htmlFor="mlflow.home.create_workspace_modal.workspace_description">
            <FormattedMessage defaultMessage="Description (optional)" description="Label for description field" />:
          </FormUI.Label>
          <RHFControlledComponents.Input
            control={form.control}
            id="mlflow.home.create_workspace_modal.workspace_description"
            componentId="mlflow.home.create_workspace_modal.workspace_description_input"
            name="workspaceDescription"
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter workspace description',
              description: 'Input placeholder for workspace description in create workspace modal',
            })}
          />
          <Spacer />
          <FormUI.Label htmlFor="mlflow.home.create_workspace_modal.workspace_artifact_root">
            <FormattedMessage
              defaultMessage="Default Artifact Root (optional)"
              description="Label for artifact root field"
            />
            :
          </FormUI.Label>
          <RHFControlledComponents.Input
            control={form.control}
            id="mlflow.home.create_workspace_modal.workspace_artifact_root"
            componentId="mlflow.home.create_workspace_modal.workspace_artifact_root_input"
            name="workspaceArtifactRoot"
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter default artifact root URI',
              description: 'Input placeholder for artifact root in create workspace modal',
            })}
          />
        </div>
      </Modal>
    </FormProvider>
  );

  const openModal = () => {
    setError(null);
    form.reset();
    setOpen(true);
  };

  return { CreateWorkspaceModal: modalElement, openModal };
};
