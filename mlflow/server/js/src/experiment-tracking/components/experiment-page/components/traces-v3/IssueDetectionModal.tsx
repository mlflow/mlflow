import React, { useState, useCallback, useRef } from 'react';
import { Modal, Button, useDesignSystemTheme, SparkleIcon, Typography, Alert } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { SelectTracesModal } from '../../../SelectTracesModal';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { ALL_ISSUE_CATEGORIES, type IssueCategory } from './IssueDetectionCategories';
import { IssueDetectionCategorySelection } from './IssueDetectionCategorySelection';
import { IssueDetectionModelSelection, type IssueDetectionModelSelectionRef } from './IssueDetectionModelSelection';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';

interface IssueDetectionModalProps {
  onClose: () => void;
  experimentId?: string;
  initialSelectedTraceIds?: string[];
  availableTraceIds?: string[];
  onSubmitSuccess?: (runId?: string) => void;
}

export const IssueDetectionModal: React.FC<IssueDetectionModalProps> = ({
  onClose,
  experimentId,
  initialSelectedTraceIds = [],
  availableTraceIds = [],
  onSubmitSuccess,
}) => {
  const { theme } = useDesignSystemTheme();
  const modelSelectionRef = useRef<IssueDetectionModelSelectionRef>(null);

  const [currentStep, setCurrentStep] = useState<1 | 2>(1);
  const [selectedCategories, setSelectedCategories] = useState<Set<IssueCategory>>(new Set(ALL_ISSUE_CATEGORIES));
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>(() => {
    return initialSelectedTraceIds.length > 0 ? initialSelectedTraceIds : availableTraceIds;
  });
  const [isSelectTracesModalOpen, setIsSelectTracesModalOpen] = useState(false);
  const [isModelSelectionValid, setIsModelSelectionValid] = useState(false);

  const {
    mutate: createSecret,
    isLoading: isCreatingSecret,
    error: createSecretError,
    reset: resetCreateSecret,
  } = useCreateSecret();

  const {
    mutate: invokeIssueDetection,
    isLoading: isInvokingIssueDetection,
    error: issueDetectionError,
    reset: resetIssueDetection,
  } = useInvokeIssueDetection();

  const resetForm = useCallback(() => {
    setCurrentStep(1);
    setSelectedCategories(new Set(ALL_ISSUE_CATEGORIES));
    setSelectedTraceIds([]);
    setIsModelSelectionValid(false);
    modelSelectionRef.current?.reset();
  }, []);

  const handleCategoryToggle = useCallback((categoryId: IssueCategory, isChecked: boolean) => {
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      if (isChecked) {
        next.add(categoryId);
      } else {
        next.delete(categoryId);
      }
      return next;
    });
  }, []);

  const handleNext = useCallback(() => {
    setCurrentStep(2);
  }, []);

  const handlePrevious = useCallback(() => {
    setCurrentStep(1);
  }, []);

  const handleSubmit = () => {
    const values = modelSelectionRef.current?.getValues();
    if (!values || !experimentId) return;

    const { provider, model, apiKeyConfig, saveKey } = values;

    const submitIssueDetection = (secretId: string) => {
      invokeIssueDetection(
        {
          experimentId,
          traceIds: selectedTraceIds,
          categories: Array.from(selectedCategories),
          provider,
          model,
          secret_id: secretId,
        },
        {
          onSuccess: (response) => {
            onSubmitSuccess?.(response.run_id);
            resetForm();
            onClose();
          },
        },
      );
    };

    if (saveKey && apiKeyConfig.mode === 'new') {
      const authConfig = { ...apiKeyConfig.newSecret.configFields } satisfies Record<string, string>;
      if (apiKeyConfig.newSecret.authMode) {
        authConfig['auth_mode'] = apiKeyConfig.newSecret.authMode;
      }

      createSecret(
        {
          secret_name: apiKeyConfig.newSecret.name,
          secret_value: apiKeyConfig.newSecret.secretFields,
          provider: provider,
          auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
        },
        {
          onSuccess: (response) => {
            submitIssueDetection(response.secret.secret_id);
          },
        },
      );
    } else if (apiKeyConfig.mode === 'existing' && apiKeyConfig.existingSecretId) {
      submitIssueDetection(apiKeyConfig.existingSecretId);
    }
  };

  const handleClose = useCallback(() => {
    resetForm();
    resetCreateSecret();
    resetIssueDetection();
    onClose();
  }, [resetForm, resetCreateSecret, resetIssueDetection, onClose]);

  const isStep1Valid = selectedCategories.size > 0;
  const isStep2Valid = isModelSelectionValid;

  const handleModelSelectionValidityChange = useCallback((isValid: boolean) => {
    setIsModelSelectionValid(isValid);
  }, []);

  const renderStep1Footer = () => (
    <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
      <Button componentId="mlflow.traces.issue-detection-modal.cancel" onClick={handleClose}>
        <FormattedMessage defaultMessage="Cancel" description="Cancel button in issue detection modal" />
      </Button>
      <Button
        componentId="mlflow.traces.issue-detection-modal.next"
        type="primary"
        onClick={handleNext}
        disabled={!isStep1Valid}
      >
        <FormattedMessage defaultMessage="Next" description="Next button to proceed to provider configuration" />
      </Button>
    </div>
  );

  const renderStep2Footer = () => (
    <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
      <Button componentId="mlflow.traces.issue-detection-modal.previous" onClick={handlePrevious}>
        <FormattedMessage defaultMessage="Previous" description="Previous button to go back to category selection" />
      </Button>
      <Button
        componentId="mlflow.traces.issue-detection-modal.submit"
        type="primary"
        onClick={handleSubmit}
        loading={isCreatingSecret || isInvokingIssueDetection}
        disabled={!isStep2Valid}
      >
        <SparkleIcon css={{ marginRight: theme.spacing.xs }} />
        <FormattedMessage defaultMessage="Run Analysis" description="Submit button to trigger issue detection job" />
      </Button>
    </div>
  );

  return (
    <>
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
        visible
        onCancel={handleClose}
        footer={currentStep === 1 ? renderStep1Footer() : renderStep2Footer()}
      >
        {(createSecretError || issueDetectionError) && (
          <Alert
            componentId="mlflow.traces.issue-detection-modal.error"
            type="error"
            message={createSecretError?.message || issueDetectionError?.message}
            closable
            onClose={() => {
              resetCreateSecret();
              resetIssueDetection();
            }}
            css={{ marginBottom: theme.spacing.md }}
          />
        )}
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.lg }}>
          <FormattedMessage
            defaultMessage="Use AI to automatically analyze your traces and identify potential issues"
            description="Description text for issue detection modal"
          />
        </Typography.Text>
        {currentStep === 1 ? (
          <IssueDetectionCategorySelection
            selectedCategories={selectedCategories}
            onCategoryToggle={handleCategoryToggle}
          />
        ) : (
          <IssueDetectionModelSelection
            ref={modelSelectionRef}
            selectedTraceIds={selectedTraceIds}
            onSelectTracesClick={() => setIsSelectTracesModalOpen(true)}
            onValidityChange={handleModelSelectionValidityChange}
          />
        )}
      </Modal>
      {isSelectTracesModalOpen && (
        <SelectTracesModal
          onClose={() => setIsSelectTracesModalOpen(false)}
          onSuccess={(traceIds) => {
            setSelectedTraceIds(traceIds);
            setIsSelectTracesModalOpen(false);
          }}
          initialTraceIdsSelected={selectedTraceIds}
        />
      )}
    </>
  );
};
