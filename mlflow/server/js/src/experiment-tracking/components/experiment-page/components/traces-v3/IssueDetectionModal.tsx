import React, { useState, useCallback } from 'react';
import { Modal, Button, useDesignSystemTheme, SparkleIcon, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ProviderSelect } from '../../../../../gateway/components/create-endpoint/ProviderSelect';

interface IssueDetectionModalProps {
  visible: boolean;
  onClose: () => void;
  experimentId?: string;
}

const DEFAULT_MODELS_BY_PROVIDER: Record<string, { analysisModel: string; judgeModel: string }> = {
  openai: { analysisModel: 'gpt-5', judgeModel: 'gpt-5-mini' },
  anthropic: { analysisModel: 'claude-sonnet-4-20250514', judgeModel: 'claude-haiku-4-20250514' },
  databricks: { analysisModel: 'databricks-gpt-5', judgeModel: 'databricks-gpt-5-mini' },
};

export const IssueDetectionModal: React.FC<IssueDetectionModalProps> = ({ visible, onClose, experimentId }) => {
  const { theme } = useDesignSystemTheme();
  const [provider, setProvider] = useState('');
  const [analysisModel, setAnalysisModel] = useState('');
  const [judgeModel, setJudgeModel] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleProviderChange = useCallback((newProvider: string) => {
    setProvider(newProvider);
    const defaults = DEFAULT_MODELS_BY_PROVIDER[newProvider];
    setAnalysisModel(defaults?.analysisModel ?? '');
    setJudgeModel(defaults?.judgeModel ?? '');
  }, []);

  const resetForm = useCallback(() => {
    setProvider('');
    setAnalysisModel('');
    setJudgeModel('');
  }, []);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      // TODO: Implement backend API call for issue detection
      // eslint-disable-next-line no-console
      console.log('Issue detection triggered:', {
        provider,
        analysisModel,
        judgeModel,
        experimentId,
      });
      resetForm();
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = useCallback(() => {
    resetForm();
    onClose();
  }, [resetForm, onClose]);

  const isSubmitDisabled = !provider || !analysisModel || !judgeModel;

  return (
    <Modal
      componentId="mlflow.traces.issue-detection-modal"
      title={
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <SparkleIcon color="ai" />
          <FormattedMessage
            defaultMessage="Detect Issues"
            description="Title of the issue detection configuration modal"
          />
        </div>
      }
      visible={visible}
      onCancel={handleClose}
      footer={
        <Button
          componentId="mlflow.traces.issue-detection-modal.submit"
          type="primary"
          onClick={handleSubmit}
          loading={isSubmitting}
          disabled={isSubmitDisabled}
          css={{ width: '100%' }}
        >
          <SparkleIcon css={{ marginRight: theme.spacing.xs }} />
          <FormattedMessage defaultMessage="Run Analysis" description="Submit button to trigger issue detection job" />
        </Button>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Connect an LLM to run an AI-powered issue analysis on your traces"
            description="Description text for issue detection modal"
          />
        </Typography.Text>

        <div>
          <ProviderSelect
            value={provider}
            onChange={handleProviderChange}
            componentIdPrefix="mlflow.traces.issue-detection-modal.provider"
          />
          {provider && DEFAULT_MODELS_BY_PROVIDER[provider] && (
            <Typography.Text
              color="secondary"
              css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
            >
              <FormattedMessage
                defaultMessage="Analysis model: {analysisModel} · Judge model: {judgeModel}"
                description="Display of default models for selected provider"
                values={{
                  analysisModel,
                  judgeModel,
                }}
              />
            </Typography.Text>
          )}
          {provider && !DEFAULT_MODELS_BY_PROVIDER[provider] && (
            <Typography.Text
              color="secondary"
              css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
            >
              <FormattedMessage
                defaultMessage="Please select models in `Advanced settings` below"
                description="Message when provider has no default models"
              />
            </Typography.Text>
          )}
        </div>
      </div>
    </Modal>
  );
};
