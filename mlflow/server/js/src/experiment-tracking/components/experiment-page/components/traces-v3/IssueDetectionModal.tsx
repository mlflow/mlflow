import React, { useEffect, useState, useCallback, useRef } from 'react';
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
import { useIssueDetectionSettings } from '../../../../pages/experiment-issues/useIssueDetectionSettings';
import {
  defaultIssueDetectionSettings,
  type IssueDetectionSettings,
  noFocusAreaSelected,
  selectedDetectionCategories,
} from '../../../../pages/experiment-issues/issueDetectionSettings';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';
import { recordSubmittedIssueDetectionJob } from './IssueDetectionJobNotifications';
import { estimateIssueDetectionCostUsd, formatEstimatedCostUsd } from './issueDetectionCostEstimate';
import {
  ISSUE_DETECTION_PROVIDERS,
  IssueDetectionModelDropdown,
  type IssueDetectionModelSelection,
} from './IssueDetectionModelDropdown';
import { useActiveWorkspace } from '../../../../../workspaces/utils/WorkspaceUtils';
import { useExperimentPermission } from '../../../../hooks/useExperimentPermission';
import heroImg from '../../../../../common/static/issue-detection-empty.svg';
import { IssueDetectionSettingsForm } from '../../../../pages/experiment-issues/IssueDetectionSettingsForm';
import {
  IssueDetectionTraceScope,
  useNormalizeIssueDetectionTraceScope,
} from '../../../../pages/experiment-issues/IssueDetectionTraceScope';
import {
  compileIssueDetectionTraceScope,
  emptyIssueDetectionTraceScope,
  type IssueDetectionTraceScopeValue,
} from '../../../../pages/experiment-issues/issueDetectionTraceScope';
import { useIssueDetectionTraceScope } from '../../../../pages/experiment-issues/useIssueDetectionTraceScope';
import { useHasExperimentIssues } from '../../../../pages/experiment-issues/useExperimentIssues';
import { searchMlflowTracesQueryFn } from '../../../../../shared/web-shared/genai-traces-table/hooks/useMlflowTraces';
import { escapeFilterValue } from '../traces-v4/utils/buildTracesV4SearchParams';

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

type ModalView = 'main' | 'apiKey' | 'customize';

export const IssueDetectionModal: React.FC<IssueDetectionModalProps> = ({
  onClose,
  experimentId,
  initialSelectedTraceIds = [],
  availableTraceIds = [],
  defaultGroupBySession = false,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const workspace = useActiveWorkspace();

  // Without an explicit table selection, default to the most recent traces
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>(() => {
    return initialSelectedTraceIds.length > 0
      ? initialSelectedTraceIds
      : availableTraceIds.slice(0, QUICK_SELECT_TRACE_COUNT);
  });
  const [explicitSelection, setExplicitSelection] = useState(initialSelectedTraceIds.length > 0);
  const [selection, setSelection] = useState<IssueDetectionModelSelection | null>(null);
  const [view, setView] = useState<ModalView>('main');
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  const [isSelectTracesModalOpen, setIsSelectTracesModalOpen] = useState(false);

  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();

  // Read here rather than taking them as props, so every entry point into detection -- the
  // Issues tab and both Traces tabs -- honours the same saved settings.
  const {
    data: detectionSettings,
    isLoading: isLoadingDetectionSettings,
    error: detectionSettingsError,
    refetch: refetchSettings,
    save: saveSettings,
  } = useIssueDetectionSettings(experimentId ?? '', Boolean(experimentId));
  const scope = useIssueDetectionTraceScope(experimentId ?? '', Boolean(experimentId));
  const normalizeScope = useNormalizeIssueDetectionTraceScope();
  const canEdit = useExperimentPermission(experimentId ?? '', ['EDIT', 'MANAGE']);
  const [configuredSettings, setConfiguredSettings] = useState<IssueDetectionSettings>();
  const [configuredScope, setConfiguredScope] = useState<IssueDetectionTraceScopeValue>();
  const [settingsDraft, setSettingsDraft] = useState<IssueDetectionSettings>(defaultIssueDetectionSettings);
  const [scopeDraft, setScopeDraft] = useState<IssueDetectionTraceScopeValue>(emptyIssueDetectionTraceScope);
  const [isPreparing, setIsPreparing] = useState(false);
  const preparing = useRef(false);
  const [configurationError, setConfigurationError] = useState<Error>();
  const history = useHasExperimentIssues(experimentId ?? '', Boolean(experimentId));
  const [onboardingResolved, setOnboardingResolved] = useState(false);
  const effectiveSettings = (canEdit ? configuredSettings : undefined) ?? detectionSettings;
  const effectiveScope = (canEdit ? configuredScope : undefined) ?? scope.data ?? emptyIssueDetectionTraceScope();
  const configurationLoadError = detectionSettingsError ?? scope.error;
  // A run submitted before the settings arrive would silently use the defaults and drop the
  // saved priorities, which is worse than waiting: it is an expensive run whose results
  // quietly ignore the experiment's configuration.
  const isWaitingForDetectionSettings = Boolean(
    experimentId &&
    ((isLoadingDetectionSettings && !detectionSettings && !detectionSettingsError) ||
      (scope.isLoading && !scope.data && !scope.error)),
  );
  // The one case where what is stored and what is sent differ, so it is disclosed here as well
  // as in the settings modal: this is where the run is actually started, and someone who saved
  // the setting a week ago is not looking at that modal now.
  const savedSettingsSelectNothing = noFocusAreaSelected(effectiveSettings);

  useEffect(() => {
    if (onboardingResolved || isWaitingForDetectionSettings || configurationLoadError || history.isLoading) return;
    // A completed run with no issues is still a first run for setup purposes.
    if (!history.data?.issues?.length) {
      setSettingsDraft(detectionSettings ?? defaultIssueDetectionSettings());
      setScopeDraft(scope.data ?? emptyIssueDetectionTraceScope());
      setView('customize');
    }
    setOnboardingResolved(true);
  }, [
    onboardingResolved,
    isWaitingForDetectionSettings,
    configurationLoadError,
    history.isLoading,
    history.data,
    detectionSettings,
    scope.data,
  ]);

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
  } = useInvokeIssueDetection(experimentId);

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
    !isWaitingForDetectionSettings &&
    !configurationLoadError &&
    onboardingResolved &&
    !isPreparing &&
    !isInvokingIssueDetection &&
    Boolean(
      selection && (selection.mode === 'endpoint' ? selection.endpointName : selection.provider && selection.model),
    );

  const submitRun = async (secretIdOverride?: string) => {
    if (!selection || !experimentId || !isFormValid || preparing.current) return;
    preparing.current = true;
    setIsPreparing(true);
    setConfigurationError(undefined);
    try {
      if (canEdit && configuredSettings && JSON.stringify(configuredSettings) !== JSON.stringify(detectionSettings)) {
        await saveSettings.mutateAsync(configuredSettings);
      }
      if (canEdit && configuredScope && JSON.stringify(configuredScope) !== JSON.stringify(scope.data)) {
        await scope.save.mutateAsync(configuredScope);
      }
      let traceIds = selectedTraceIds;
      const savedFilter = compileIssueDetectionTraceScope(effectiveScope);
      if (savedFilter) {
        const selectionFilter = explicitSelection
          ? `attributes.request_id IN (${selectedTraceIds.map((id) => `'${escapeFilterValue(id)}'`).join(',')})`
          : undefined;
        const traces = await searchMlflowTracesQueryFn({
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
          filter: [selectionFilter, savedFilter].filter(Boolean).join(' AND '),
          pageSize: explicitSelection ? Math.min(selectedTraceIds.length, 500) : QUICK_SELECT_TRACE_COUNT,
          limit: explicitSelection ? selectedTraceIds.length : QUICK_SELECT_TRACE_COUNT,
          orderBy: ['timestamp DESC'],
        });
        traceIds = traces.map((trace) => trace.trace_id);
        if (!traceIds.length) {
          throw new Error(
            'No selected traces match the saved trace scope. Change the selection or customize the scope.',
          );
        }
      }

      invokeIssueDetection(
        {
          experimentId,
          traceIds,
          categories: selectedDetectionCategories(effectiveSettings),
          instructions: effectiveSettings?.instructions || undefined,
          provider: selection.provider,
          model: selection.model,
          secret_id: selection.mode === 'direct' ? (secretIdOverride ?? existingSecrets[0]?.secret_id) : undefined,
          endpoint_name: selection.mode === 'endpoint' ? selection.endpointName : undefined,
        },
        {
          onSuccess: (response) => {
            const traceCount = traceIds.length;
            recordSubmittedIssueDetectionJob({
              workspace,
              experimentId,
              jobId: response.job_id,
              runId: response.run_id,
              traceCount,
            });
            onClose();
          },
        },
      );
    } catch (error) {
      setConfigurationError(error instanceof Error ? error : new Error(String(error)));
    } finally {
      preparing.current = false;
      setIsPreparing(false);
    }
  };

  const handleContinueAndRun = () => {
    if (!selection || !apiKeyDraft.trim() || !isFormValid || isCreatingSecret) return;
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
      {savedSettingsSelectNothing && (
        <Alert
          componentId="mlflow.traces.issue-detection-modal.no-focus-area-selected"
          type="info"
          css={{ marginTop: theme.spacing.md }}
          message={intl.formatMessage({
            defaultMessage:
              "This experiment's issue detection settings have no focus area selected, so this run considers all of them.",
            description: 'Note in the issue detection modal shown when the saved settings deselect every focus area',
          })}
          closable={false}
        />
      )}
      {configurationLoadError && (
        <Alert
          componentId="mlflow.traces.issue-detection-modal.settings-load-error"
          type="error"
          closable={false}
          message={intl.formatMessage({
            defaultMessage: 'Could not load issue detection settings. Retry before starting analysis.',
            description: 'Blocking detection settings read error',
          })}
          actions={[
            {
              componentId: 'mlflow.traces.issue-detection-modal.retry-settings',
              children: intl.formatMessage({
                defaultMessage: 'Retry',
                description: 'Retry issue detection configuration',
              }),
              onClick: () => {
                void refetchSettings();
                void scope.refetch();
              },
            },
          ]}
        />
      )}
      <Button
        componentId="mlflow.traces.issue-detection-modal.customize"
        type="link"
        disabled={
          !canEdit ||
          isWaitingForDetectionSettings ||
          Boolean(configurationLoadError) ||
          isPreparing ||
          isInvokingIssueDetection
        }
        onClick={() => {
          setSettingsDraft(effectiveSettings ?? defaultIssueDetectionSettings());
          setScopeDraft(effectiveScope);
          setView('customize');
        }}
      >
        <FormattedMessage
          defaultMessage="Customize"
          description="Customize detection priorities and saved trace scope"
        />
      </Button>
      {compileIssueDetectionTraceScope(effectiveScope) && (
        <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="The saved trace scope further limits which traces are analyzed."
            description="Manual detection uses shared scope"
          />
        </Typography.Text>
      )}
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
    if (view === 'customize') {
      return (
        <>
          <Button componentId="mlflow.traces.issue-detection-modal.customize-cancel" onClick={() => setView('main')}>
            <FormattedMessage defaultMessage="Cancel" description="Discard customization draft" />
          </Button>
          <Button
            componentId="mlflow.traces.issue-detection-modal.customize-apply"
            type="primary"
            disabled={canEdit && noFocusAreaSelected(settingsDraft)}
            onClick={() => {
              setConfiguredSettings(settingsDraft);
              setConfiguredScope(normalizeScope(scopeDraft));
              setView('main');
            }}
          >
            <FormattedMessage defaultMessage="Apply" description="Apply customization to pending analysis" />
          </Button>
        </>
      );
    }

    if (view === 'apiKey') {
      return (
        <>
          <Button
            componentId="mlflow.traces.issue-detection-modal.api-key-back"
            disabled={isCreatingSecret || isPreparing || isInvokingIssueDetection}
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
            loading={isCreatingSecret || isPreparing || isInvokingIssueDetection}
            disabled={!apiKeyDraft.trim() || !isFormValid}
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
            loading={isPreparing || isInvokingIssueDetection}
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
        width: view === 'customize' ? 720 : 520,
        bodyStyle: { paddingLeft: 32, paddingRight: 32, paddingBottom: 24, overflowY: 'auto' },
      }}
      onCancel={
        isInvokingIssueDetection || isCreatingSecret || isPreparing
          ? undefined
          : view === 'customize'
            ? () => setView('main')
            : handleClose
      }
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
        {configurationError && (
          <Alert
            componentId="mlflow.traces.issue-detection-modal.configuration-error"
            type="error"
            message={configurationError.message}
            closable={false}
          />
        )}
        {view === 'main' && renderMainView()}
        {view === 'customize' && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            {history.error && (
              <Alert
                componentId="mlflow.traces.issue-detection-modal.history-unavailable"
                type="info"
                message={intl.formatMessage({
                  defaultMessage: 'Issue history is unavailable. You can still configure and run analysis.',
                  description: 'Advisory issue history read error',
                })}
                closable={false}
              />
            )}
            <IssueDetectionSettingsForm draft={settingsDraft} setDraft={setSettingsDraft} disabled={!canEdit} />
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Filter target traces"
                description="Detection setup trace scope section"
              />
            </Typography.Text>
            <IssueDetectionTraceScope scope={scopeDraft} onChange={setScopeDraft} disabled={!canEdit} />
          </div>
        )}

        {view === 'apiKey' && renderApiKeyView()}
        {isSelectTracesModalOpen && (
          <SelectTracesModal
            onClose={() => setIsSelectTracesModalOpen(false)}
            onSuccess={(traceIds) => {
              setExplicitSelection(true);
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
