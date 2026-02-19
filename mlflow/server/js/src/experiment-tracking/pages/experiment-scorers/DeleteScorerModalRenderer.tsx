import React from 'react';
import { DangerModal, Alert, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import type { ScheduledScorer } from './types';
import type { PredefinedError } from '@databricks/web-shared/errors';
import { COMPONENT_ID_PREFIX } from './constants';

interface DeleteScorerModalRendererProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  scorer: ScheduledScorer;
  isLoading?: boolean;
  error?: PredefinedError | null;
}

export const DeleteScorerModalRenderer: React.FC<DeleteScorerModalRendererProps> = ({
  isOpen,
  onClose,
  onConfirm,
  scorer,
  isLoading = false,
  error,
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <DangerModal
      componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_deletescorermodalrenderer_28"
      title={
        <FormattedMessage defaultMessage="Delete judge" description="Title for the delete judge confirmation modal" />
      }
      visible={isOpen}
      onCancel={onClose}
      onOk={onConfirm}
      confirmLoading={isLoading}
    >
      <>
        <FormattedMessage
          defaultMessage="Are you sure you want to delete the judge ''{scorerName}''? This action cannot be undone."
          description="Confirmation message for deleting a judge"
          values={{ scorerName: scorer.name }}
        />
        {error && (
          <Alert
            componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_deletescorermodalrenderer_46"
            type="error"
            message={error.message || error.displayMessage}
            closable={false}
            css={{ marginTop: theme.spacing.md }}
          />
        )}
      </>
    </DangerModal>
  );
};
