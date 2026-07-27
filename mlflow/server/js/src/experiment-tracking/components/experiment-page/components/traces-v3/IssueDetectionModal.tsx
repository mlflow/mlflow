import React, { useEffect, useState, useCallback } from 'react';
import {
  Modal,
  Button,
  Input,
  LightbulbIcon,
  PencilIcon,
  useDesignSystemTheme,
  SparkleIcon,
  Tooltip,
  Typography,
  Alert,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { generateRandomName } from '../../../../../common/utils/NameUtils';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { SelectTracesModal } from '../../../SelectTracesModal';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { ALL_ISSUE_CATEGORIES } from './IssueDetectionCategories';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';
import { recordSubmittedIssueDetectionJob } from './IssueDetectionJobNotifications';
import { estimateIssueDetectionCostUsd, formatEstimatedCostUsd } from './issueDetectionCostEstimate';
import {
  ISSUE_DETECTION_PROVIDERS,
  IssueDetectionModelDropdown,
  type IssueDetectionModelSelection,
} from './IssueDetectionModelDropdown';
import heroImg from '../../../../../common/static/issue-detection-empty.svg';

interface IssueDetectionModalProps {
  onClose: () => void;
  experimentId?: string;
  initialSelectedTraceIds?: string[];
  availableTraceIds?: string[];
  defaultGroupBySession?: boolean;
}

const QUICK_SELECT_TRACE_COUNT = 50;
const MIN_RECOMMENDED_TRACE_COUNT = 10;

const MISSING_API_KEY_ERROR_FRAGMENT = 'No API key available';

type ModalView = 'main' | 'apiKey';

export const IssueDetectionModal: React.FC<IssueDetectionModalProps> = ({
  onClose,
  experimentId,
  initialSelectedTraceIds = [],
  availableTraceIds = [],
  defaultGroupBySession = false,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Without an explicit table selection, default to the most recent traces
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>(() => {
    return initialSelectedTraceIds.length > 0
      ? initialSelectedTraceIds
      : availableTraceIds.slice(0, QUICK_SELECT_TRACE_COUNT);
  });
  const [selection, setSelection] = useState<IssueDetectionModelSelection | null>(null);
  const [view, setView] = useState<ModalView>('main');
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  const [isSelectTracesModalOpen, setIsSelectTracesModalOpen] = useState(false);

  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();

  // Default to the first gateway endpoint, else the first core provider
  useEffect(() => {
    if (!selection && !isLoadingEndpoints) {
      const provider = ISSUE_DETECTION_PROVIDERS[0];
      if (endpoints.length > 0) {
        setSelection({
          mode: 'endpoint',
          endpointName: endpoints[0].name,
          provider: provider.id,
          model: provider.defaultModel,
        });
      } else {
        setSelection({ mode: 'direct', provider: provider.id, model: provider.defaultModel });
      }
    }
  }, [selection, isLoadingEndpoints, endpoints]);

  // Direct providers use the API key already saved in AI Gateway (never asked upfront)
  const { existingSecrets } = useApiKeyConfiguration({
    provider: selection?.mode === 'direct' ? selection.provider : ISSUE_DETECTION_PROVIDERS[0].id,
  });

  const hasNoTraces = selectedTraceIds.length === 0 && availableTraceIds.length === 0;
  const showLowTraceWarning = selectedTraceIds.length > 0 && selectedTraceIds.length < MIN_RECOMMENDED_TRACE_COUNT;
  const estimatedCost = estimateIssueDetectionCostUsd(selectedTraceIds.length);

  const {
    mutate: invokeIssueDetection,
    isLoading: isInvokingIssueDetection,
    error: issueDetectionError,
    reset: resetIssueDetection,
  } = useInvokeIssueDetection();

  const {
    mutate: createSecret,
    isLoading: isCreatingSecret,
    error: createSecretError,
    reset: resetCreateSecret,
  } = useCreateSecret();

  // The server rejects keyless submissions upfront; turn that into the API key step
  const isMissingKeyError = Boolean(issueDetectionError?.message.includes(MISSING_API_KEY_ERROR_FRAGMENT));
  useEffect(() => {
    if (isMissingKeyError) {
      setView('apiKey');
      resetIssueDetection();
    }
  }, [isMissingKeyError, resetIssueDetection]);

  const isFormValid =
    selectedTraceIds.length > 0 &&
    Boolean(
      selection && (selection.mode === 'endpoint' ? selection.endpointName : selection.provider && selection.model),
    );

  const submitRun = (secretIdOverride?: string) => {
    if (!selection || !experimentId) return;

    invokeIssueDetection(
      {
        experimentId,
        traceIds: selectedTraceIds,
        categories: ALL_ISSUE_CATEGORIES,
        provider: selection.provider,
        model: selection.model,
        secret_id: selection.mode === 'direct' ? (secretIdOverride ?? existingSecrets[0]?.secret_id) : undefined,
        endpoint_name: selection.mode === 'endpoint' ? selection.endpointName : undefined,
      },
      {
        onSuccess: (response) => {
          const traceCount = selectedTraceIds.length;
          onClose();
          recordSubmittedIssueDetectionJob({
            experimentId,
            jobId: response.job_id,
            runId: response.run_id,
            traceCount,
          });
        },
      },
    );
  };

  const handleContinueAndRun = () => {
    if (!selection || !apiKeyDraft.trim()) return;
    createSecret(
      {
        secret_name: generateRandomName(selection.provider),
        secret_value: { api_key: apiKeyDraft.trim() },
        provider: selection.provider,
      },
      {
        onSuccess: (response) => {
          submitRun(response.secret.secret_id);
        },
      },
    );
  };

  const handleClose = useCallback(() => {
    resetIssueDetection();
    resetCreateSecret();
    onClose();
  }, [resetIssueDetection, resetCreateSecret, onClose]);

  const selectedDirectOption =
    selection?.mode === 'direct' ? ISSUE_DETECTION_PROVIDERS.find((option) => option.id === selection.provider) : null;

  const summaryCardCss = {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
    padding: theme.spacing.sm,
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.borders.borderRadiusMd,
    cursor: 'pointer',
    '&:hover': {
      backgroundColor: theme.colors.actionTertiaryBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
    },
  } as const;

  const renderMainView = () => (
    <>
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
        <img
          src={heroImg}
          alt={intl.formatMessage({
            defaultMessage: 'Illustration of traces being analyzed for issues',
            description: 'Alt text for the issue detection illustration',
          })}
          css={{ maxWidth: '100%', maxHeight: 120 }}
        />
        <Typography.Text css={{ marginTop: theme.spacing.md }}>
          <FormattedMessage
            defaultMessage="Find failure patterns hiding in your traces, automatically."
            description="Headline for the issue detection modal"
          />
        </Typography.Text>
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="AI reviews every trace, groups failures into issues, and shows you what to fix. No manual trace reading required."
            description="Supporting description for the issue detection modal"
          />
        </Typography.Text>
      </div>
      <div
        css={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: theme.spacing.md,
          marginTop: theme.spacing.md,
          paddingTop: theme.spacing.sm,
          borderTop: `1px solid ${theme.colors.border}`,
        }}
      >
        <div css={{ flex: 1, minWidth: 0 }}>
          <Typography.Text bold color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Model"
              description="Column header for the model powering issue detection"
            />
          </Typography.Text>
          {selection && <IssueDetectionModelDropdown endpoints={endpoints} value={selection} onChange={setSelection} />}
        </div>
        <div css={{ flex: 1, minWidth: 0 }}>
          <Typography.Text bold color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage defaultMessage="Traces" description="Column header for the analyzed traces" />
          </Typography.Text>
          {hasNoTraces ? (
            <Typography.Text css={{ display: 'block' }} color="secondary">
              <FormattedMessage
                defaultMessage="No traces yet. Log traces to this experiment first."
                description="Message shown in the issue detection modal when the experiment has no traces"
              />
            </Typography.Text>
          ) : (
            <div
              role="button"
              tabIndex={0}
              data-testid="traces-card"
              onClick={() => setIsSelectTracesModalOpen(true)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') setIsSelectTracesModalOpen(true);
              }}
              css={summaryCardCss}
            >
              <div css={{ minWidth: 0, flex: 1, textAlign: 'left' }}>
                <Typography.Text css={{ display: 'block' }}>
                  <FormattedMessage
                    defaultMessage="{count, plural, one {1 trace selected} other {# traces selected}}"
                    description="Label showing number of traces selected"
                    values={{ count: selectedTraceIds.length }}
                  />
                </Typography.Text>
                {selectedTraceIds.length > 0 && (
                  <Typography.Hint>
                    <FormattedMessage
                      defaultMessage="Estimated cost: ~{low}-{high}"
                      description="Estimated USD cost range for the issue detection run"
                      values={{
                        low: formatEstimatedCostUsd(estimatedCost.low),
                        high: formatEstimatedCostUsd(estimatedCost.high),
                      }}
                    />
                  </Typography.Hint>
                )}
              </div>
              <PencilIcon css={{ color: theme.colors.textSecondary }} />
            </div>
          )}
        </div>
      </div>
      {showLowTraceWarning && (
        <div
          data-testid="low-trace-warning"
          css={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: theme.spacing.sm,
            marginTop: theme.spacing.md,
            padding: theme.spacing.sm,
            backgroundColor: theme.colors.backgroundWarning,
            border: `1px solid ${theme.colors.borderWarning}`,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          <LightbulbIcon css={{ color: theme.colors.textValidationWarning, marginTop: 2 }} />
          <Typography.Text>
            <FormattedMessage
              defaultMessage="You selected only {count, plural, one {1 trace} other {# traces}}. Analyze at least {recommended} for more accurate results."
              description="Tip shown when fewer than the recommended number of traces are selected"
              values={{ count: selectedTraceIds.length, recommended: MIN_RECOMMENDED_TRACE_COUNT }}
            />
          </Typography.Text>
        </div>
      )}
    </>
  );

  const renderApiKeyView = () => (
    <div data-testid="api-key-view" css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Text bold css={{ textAlign: 'center', marginTop: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="One last step to run issue detection"
          description="Headline of the API key step in the issue detection modal"
        />
      </Typography.Text>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="{provider} needs an API key. Paste it once and MLflow saves it securely in AI Gateway for all future runs."
            description="Explanation of the API key step in the issue detection modal"
            values={{ provider: selectedDirectOption?.name ?? selection?.provider }}
          />
        </Typography.Text>
      </div>
      <Input
        componentId="mlflow.traces.issue-detection-modal.api-key-input"
        data-testid="api-key-input"
        type="password"
        value={apiKeyDraft}
        onChange={(e) => setApiKeyDraft(e.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'API key',
          description: 'Placeholder of the API key input in the issue detection modal',
        })}
      />
      {createSecretError && (
        <Alert
          componentId="mlflow.traces.issue-detection-modal.error"
          type="error"
          message={createSecretError.message}
          closable
          onClose={() => resetCreateSecret()}
        />
      )}
    </div>
  );

  const renderFooter = () => {
    if (view === 'apiKey') {
      return (
        <>
          <Button
            componentId="mlflow.traces.issue-detection-modal.api-key-back"
            onClick={() => {
              resetCreateSecret();
              setView('main');
            }}
          >
            <FormattedMessage defaultMessage="Back" description="Button to go back from the API key step" />
          </Button>
          <Button
            componentId="mlflow.traces.issue-detection-modal.api-key-continue"
            type="primary"
            onClick={handleContinueAndRun}
            loading={isCreatingSecret || isInvokingIssueDetection}
            disabled={!apiKeyDraft.trim()}
          >
            <SparkleIcon css={{ marginRight: theme.spacing.xs }} />
            <FormattedMessage
              defaultMessage="Continue and run"
              description="Button to save the API key and start issue detection"
            />
          </Button>
        </>
      );
    }
    return (
      <Tooltip
        componentId="mlflow.traces.issue-detection-modal.submit.tooltip"
        content={
          isFormValid ? null : hasNoTraces ? (
            <FormattedMessage
              defaultMessage="No traces to analyze. Log traces to this experiment first."
              description="Tooltip on the disabled Run button when the experiment has no traces"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Select traces to analyze first."
              description="Tooltip on the disabled Run button when no traces are selected"
            />
          )
        }
      >
        <span css={{ display: 'inline-block' }}>
          <Button
            componentId="mlflow.traces.issue-detection-modal.submit"
            type="primary"
            onClick={() => submitRun()}
            loading={isInvokingIssueDetection}
            disabled={!isFormValid}
          >
            <SparkleIcon css={{ marginRight: theme.spacing.xs }} />
            <FormattedMessage
              defaultMessage="Run Analysis"
              description="Submit button to trigger issue detection job"
            />
          </Button>
        </span>
      </Tooltip>
    );
  };

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
      visible
      dangerouslySetAntdProps={{
        width: 520,
        bodyStyle: { paddingLeft: 32, paddingRight: 32, paddingBottom: 24, overflowY: 'auto' },
      }}
      onCancel={isInvokingIssueDetection || isCreatingSecret ? undefined : handleClose}
      footer={renderFooter()}
    >
      <div>
        {issueDetectionError && !isMissingKeyError && (
          <Alert
            componentId="mlflow.traces.issue-detection-modal.error"
            type="error"
            message={issueDetectionError.message}
            closable
            onClose={() => resetIssueDetection()}
            css={{ marginBottom: theme.spacing.md }}
          />
        )}
        {view === 'main' && renderMainView()}
        {view === 'apiKey' && renderApiKeyView()}
        {isSelectTracesModalOpen && (
          <SelectTracesModal
            onClose={() => setIsSelectTracesModalOpen(false)}
            onSuccess={(traceIds) => {
              setSelectedTraceIds(traceIds);
              setIsSelectTracesModalOpen(false);
            }}
            initialTraceIdsSelected={selectedTraceIds}
            defaultGroupBySession={defaultGroupBySession}
          />
        )}
      </div>
    </Modal>
  );
};
