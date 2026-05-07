import { useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import { Alert, FormUI, Modal, RHFControlledComponents, Spacer } from '@databricks/design-system';
import { fetchAPI, getAjaxUrl, HTTPMethods } from '../../common/utils/FetchUtils';
import { validateWorkspaceName } from '../../workspaces/utils/WorkspaceUtils';
import type { WorkspaceTraceArchivalConfigInput } from '../../workspaces/types';
import {
  DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT,
  formatTraceArchivalRetention,
  getTraceArchivalRetentionValidationError,
  type TraceArchivalRetentionUnit,
} from '../../common/utils/traceArchival';
import { WorkspaceSettingsFields } from './WorkspaceSettingsFields';

type CreateWorkspaceFormData = {
  workspaceName: string;
  workspaceDescription?: string;
  workspaceArtifactRoot?: string;
  workspaceTraceArchivalLocation?: string;
};

export const useCreateWorkspaceModal = ({ onSuccess }: { onSuccess?: (workspaceName: string) => void }) => {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [traceArchivalRetentionAmount, setTraceArchivalRetentionAmount] = useState('');
  const [traceArchivalRetentionUnit, setTraceArchivalRetentionUnit] = useState<TraceArchivalRetentionUnit>(
    DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT,
  );
  const [traceArchivalRetentionError, setTraceArchivalRetentionError] = useState<string | undefined>();
  const intl = useIntl();

  const form = useForm<CreateWorkspaceFormData>({
    defaultValues: {
      workspaceName: '',
      workspaceDescription: '',
      workspaceArtifactRoot: '',
      workspaceTraceArchivalLocation: '',
    },
  });

  const updateTraceArchivalRetention = ({
    amount = traceArchivalRetentionAmount,
    unit = traceArchivalRetentionUnit,
  }: {
    amount?: string;
    unit?: TraceArchivalRetentionUnit;
  }) => {
    setTraceArchivalRetentionAmount(amount);
    setTraceArchivalRetentionUnit(unit);
    setTraceArchivalRetentionError(getTraceArchivalRetentionValidationError(amount, unit, intl));
  };

  const handleSubmit = async (values: CreateWorkspaceFormData) => {
    setError(null);
    const traceArchivalRetention = formatTraceArchivalRetention(
      traceArchivalRetentionAmount,
      traceArchivalRetentionUnit,
    );
    const traceArchivalRetentionValidationError = getTraceArchivalRetentionValidationError(
      traceArchivalRetentionAmount,
      traceArchivalRetentionUnit,
      intl,
    );
    setTraceArchivalRetentionError(traceArchivalRetentionValidationError);
    if (traceArchivalRetentionValidationError) {
      return;
    }

    setIsLoading(true);

    const requestBody: {
      name: string;
      description?: string;
      default_artifact_root?: string;
      trace_archival_config?: WorkspaceTraceArchivalConfigInput;
    } = {
      name: values.workspaceName,
    };

    if (values.workspaceDescription) {
      requestBody.description = values.workspaceDescription;
    }

    if (values.workspaceArtifactRoot) {
      requestBody.default_artifact_root = values.workspaceArtifactRoot;
    }

    const traceArchivalLocation = values.workspaceTraceArchivalLocation?.trim();
    if (traceArchivalLocation || traceArchivalRetention) {
      requestBody.trace_archival_config = {};
      if (traceArchivalLocation) {
        requestBody.trace_archival_config.location = traceArchivalLocation;
      }
      if (traceArchivalRetention) {
        requestBody.trace_archival_config.retention = traceArchivalRetention;
      }
    }

    try {
      await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/workspaces'), {
        method: HTTPMethods.POST,
        body: JSON.stringify(requestBody),
        headers: { 'X-MLFLOW-WORKSPACE': '' },
      });

      onSuccess?.(values.workspaceName);
      setOpen(false);
      form.reset();
      updateTraceArchivalRetention({ amount: '', unit: DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT });
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
        size="wide"
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
            <FormattedMessage defaultMessage="Workspace Name" description="Label for workspace name field" /> *
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
          <WorkspaceSettingsFields<CreateWorkspaceFormData>
            idPrefix="mlflow.home.create_workspace_modal.workspace_settings"
            componentId="mlflow.home.create_workspace_modal.workspace_settings"
            fieldNames={{
              description: 'workspaceDescription',
              artifactRoot: 'workspaceArtifactRoot',
              traceArchivalLocation: 'workspaceTraceArchivalLocation',
            }}
            traceArchivalRetention={{
              amount: traceArchivalRetentionAmount,
              error: traceArchivalRetentionError,
              onAmountChange: (amount) => updateTraceArchivalRetention({ amount }),
              onUnitChange: (unit) => updateTraceArchivalRetention({ unit }),
              unit: traceArchivalRetentionUnit,
            }}
          />
        </div>
      </Modal>
    </FormProvider>
  );

  const openModal = () => {
    setError(null);
    form.reset();
    updateTraceArchivalRetention({ amount: '', unit: DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT });
    setOpen(true);
  };

  return { CreateWorkspaceModal: modalElement, openModal };
};
