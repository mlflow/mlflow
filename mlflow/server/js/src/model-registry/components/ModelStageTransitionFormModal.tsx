import type { ModalProps } from '@databricks/design-system';
import {
  FormUI,
  Modal,
  RHFControlledComponents,
  Spacer,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { archiveExistingVersionToolTipText, Stages, StageTagComponents } from '../constants';
import { useForm } from 'react-hook-form';
import { FormattedMessage } from 'react-intl';
import { useEffect } from 'react';

export interface ModelStageTransitionFormModalValues {
  comment: string;
  archiveExistingVersions: boolean;
}

export enum ModelStageTransitionFormModalMode {
  RequestOrDirect,
  Approve,
  Reject,
  Cancel,
}

export const ModelStageTransitionFormModal = ({
  visible,
  onCancel,
  toStage,
  allowArchivingExistingVersions,
  transitionDescription,
  onConfirm,
  mode = ModelStageTransitionFormModalMode.RequestOrDirect,
}: {
  toStage?: string;
  transitionDescription: React.ReactNode;
  allowArchivingExistingVersions?: boolean;
  onConfirm?: (values: ModelStageTransitionFormModalValues) => void;
  mode?: ModelStageTransitionFormModalMode;
} & Pick<ModalProps, 'visible' | 'onCancel'>) => {
  const { theme } = useDesignSystemTheme();
  const form = useForm<ModelStageTransitionFormModalValues>({
    defaultValues: {
      comment: '',
      archiveExistingVersions: false,
    },
  });

  const getModalTitle = () => {
    if (mode === ModelStageTransitionFormModalMode.Approve) {
      return (
        <FormattedMessage
          defaultMessage="Approve pending request"
          description="Title for a model version stage transition modal when approving a pending request"
        />
      );
    }
    if (mode === ModelStageTransitionFormModalMode.Reject) {
      return (
        <FormattedMessage
          defaultMessage="Reject pending request"
          description="Title for a model version stage transition modal when rejecting a pending request"
        />
      );
    }
    if (mode === ModelStageTransitionFormModalMode.Cancel) {
      return (
        <FormattedMessage
          defaultMessage="Cancel pending request"
          description="Title for a model version stage transition modal when cancelling a pending request"
        />
      );
    }
    return (
      <FormattedMessage
        defaultMessage="Stage transition"
        description="Title for a model version stage transition modal"
      />
    );
  };

  // Reset form values when modal is reopened
  useEffect(() => {
    if (visible) {
      form.reset();
    }
  }, [form, visible]);

  return (
    <Modal
      title={getModalTitle()}
      componentId="mlflow.model_registry.stage_transition_modal_v2"
      visible={visible}
      onCancel={onCancel}
      okText={
        <FormattedMessage
          defaultMessage="OK"
          description="Confirmation button text on the model version stage transition request/approval modal"
        />
      }
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancellation button text on the model version stage transition request/approval modal"
        />
      }
      onOk={onConfirm && form.handleSubmit(onConfirm)}
    >
      {transitionDescription}
      <Spacer size="sm" />
      <FormUI.Label htmlFor="mlflow.model_registry.stage_transition_modal_v2.comment">Comment</FormUI.Label>
      <RHFControlledComponents.TextArea
        name="comment"
        id="mlflow.model_registry.stage_transition_modal_v2.comment"
        componentId="mlflow.model_registry.stage_transition_modal_v2.comment"
        control={form.control}
        rows={4}
      />
      <Spacer size="sm" />

      {allowArchivingExistingVersions && toStage && (
        <RHFControlledComponents.Checkbox
          name="archiveExistingVersions"
          componentId="mlflow.model_registry.stage_transition_modal_v2.archive_existing_versions"
          control={form.control}
        >
          <Tooltip
            componentId="mlflow.model_registry.stage_transition_modal_v2.archive_existing_versions.tooltip"
            content={archiveExistingVersionToolTipText(toStage)}
          >
            <span css={{ '[role=status]': { marginRight: theme.spacing.xs } }}>
              <FormattedMessage
                defaultMessage="Transition existing {currentStage} model version to {archivedStage}"
                description="Description text for checkbox for archiving existing model versions
                  in the toStage for model version stage transition request"
                values={{
                  currentStage: <span css={{ marginLeft: theme.spacing.xs }}>{StageTagComponents[toStage]}</span>,
                  archivedStage: (
                    <span css={{ marginLeft: theme.spacing.xs }}>{StageTagComponents[Stages.ARCHIVED]}</span>
                  ),
                }}
              />
            </span>
          </Tooltip>
        </RHFControlledComponents.Checkbox>
      )}
    </Modal>
  );
};
