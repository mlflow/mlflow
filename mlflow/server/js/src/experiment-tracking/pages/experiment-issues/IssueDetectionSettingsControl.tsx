import { Alert, Button, GearIcon, FormUI, Modal, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useState } from 'react';
import { useIntl } from 'react-intl';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import {
  defaultIssueDetectionSettings,
  noFocusAreaSelected,
  type IssueDetectionSettings,
} from './issueDetectionSettings';
import { useExperimentPermission } from '../../hooks/useExperimentPermission';
import { useIssueDetectionSettings } from './useIssueDetectionSettings';
import { IssueDetectionSettingsForm } from './IssueDetectionSettingsForm';
import { IssueDetectionTraceScope, useNormalizeIssueDetectionTraceScope } from './IssueDetectionTraceScope';
import {
  emptyIssueDetectionTraceScope,
  serializeIssueDetectionTraceScope,
  InvalidIssueDetectionTraceScopeError,
  type IssueDetectionTraceScopeValue,
} from './issueDetectionTraceScope';
import { useIssueDetectionTraceScope } from './useIssueDetectionTraceScope';

const COMPONENT_ID = 'mlflow.experiment-issues.detection-settings';

export interface IssueDetectionSettingsControlProps {
  experimentId: string;
  buttonText?: React.ReactNode;
}

export const IssueDetectionSettingsControl = ({ experimentId, buttonText }: IssueDetectionSettingsControlProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const {
    data: settings,
    isLoading: isLoadingSettings,
    error: settingsLoadError,
    refetch: refetchSettings,
    save: saveMutation,
  } = useIssueDetectionSettings(experimentId);
  const scope = useIssueDetectionTraceScope(experimentId);
  const normalizeScope = useNormalizeIssueDetectionTraceScope();
  const canEdit = useExperimentPermission(experimentId, ['EDIT', 'MANAGE']);
  const loadError = settingsLoadError ?? scope.error;
  const [scopeDraft, setScopeDraft] = useState<IssueDetectionTraceScopeValue>(emptyIssueDetectionTraceScope);

  const [isOpen, setIsOpen] = useState(false);
  const [configurationError, setConfigurationError] = useState<Error>();
  const [openCount, setOpenCount] = useState(0);
  const [draft, setDraft] = useState<IssueDetectionSettings>(defaultIssueDetectionSettings);
  const open = useCallback(() => {
    setDraft(settings ?? defaultIssueDetectionSettings());
    setScopeDraft(scope.data ?? emptyIssueDetectionTraceScope());
    setOpenCount((count) => count + 1);
    setConfigurationError(undefined);
    saveMutation.reset();
    scope.save.reset();
    setIsOpen(true);
  }, [saveMutation, settings, scope]);

  const close = useCallback(() => {
    if (!saveMutation.isLoading && !scope.save.isLoading) {
      setIsOpen(false);
    }
  }, [saveMutation.isLoading, scope.save.isLoading]);

  const save = useCallback(async () => {
    if (!canEdit) return;
    setConfigurationError(undefined);
    try {
      const normalized = normalizeScope(scopeDraft);
      const serializedScope = serializeIssueDetectionTraceScope(normalized);
      const scopeChanged =
        serializedScope !== serializeIssueDetectionTraceScope(scope.data ?? emptyIssueDetectionTraceScope());
      await saveMutation.mutateAsync(draft);
      if (scopeChanged) {
        try {
          await scope.save.mutateAsync(normalized);
        } catch (cause) {
          const detail =
            cause instanceof ErrorWrapper
              ? cause.getMessageField()
              : cause instanceof Error
                ? cause.message
                : undefined;
          throw new Error(
            intl.formatMessage({
              defaultMessage:
                'Priorities and focus areas were saved, but trace scope was not. Retry to finish saving, or cancel to keep the saved changes.',
              description:
                'Issue detection settings modal > Partial save error after priorities succeeded and scope failed',
            }) + (detail ? ` ${detail}` : ''),
            { cause },
          );
        }
      }
      setIsOpen(false);
    } catch (error) {
      setConfigurationError(error instanceof Error ? error : undefined);
    }
  }, [canEdit, draft, saveMutation, normalizeScope, scope, scopeDraft, intl]);

  const settingsLabel = intl.formatMessage({
    defaultMessage: 'Issue detection settings',
    description: 'Issues page > Accessible label and tooltip for the issue detection settings button',
  });

  return (
    <>
      {loadError ? (
        // Editing has to stay blocked -- saving the defaults shown against a failed read would
        // overwrite whatever is really stored -- but the retry has to live out here, because the
        // modal that would otherwise carry the message is exactly what is being withheld.
        <Tooltip
          componentId={`${COMPONENT_ID}.load-failed-tooltip`}
          content={intl.formatMessage({
            defaultMessage: "Couldn't read this experiment's issue detection settings. Retry to edit them.",
            description:
              'Issue detection settings > Tooltip explaining the settings button could not load and offers a retry',
          })}
        >
          <Button
            componentId={`${COMPONENT_ID}.retry-load`}
            icon={<GearIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Retry loading issue detection settings',
              description:
                'Issue detection settings > Accessible label of the button that retries a failed settings read',
            })}
            onClick={() => {
              void refetchSettings();
              void scope.refetch();
            }}
          />
        </Tooltip>
      ) : (
        <Tooltip componentId={`${COMPONENT_ID}.tooltip`} content={settingsLabel}>
          <Button
            componentId={`${COMPONENT_ID}.open`}
            icon={<GearIcon />}
            aria-label={buttonText ? undefined : settingsLabel}
            disabled={!canEdit || isLoadingSettings || scope.isLoading}
            onClick={open}
          >
            {buttonText}
          </Button>
        </Tooltip>
      )}
      {scope.error instanceof InvalidIssueDetectionTraceScopeError && (
        <Alert
          componentId={`${COMPONENT_ID}.invalid-scope`}
          type="error"
          closable={false}
          message={
            scope.save.error
              ? intl.formatMessage({
                  defaultMessage: 'Could not reset trace scope. Try again.',
                  description: 'Invalid issue detection trace scope reset error',
                })
              : scope.error.message
          }
          actions={
            canEdit
              ? [
                  {
                    componentId: `${COMPONENT_ID}.reset-scope`,
                    children: intl.formatMessage({
                      defaultMessage: 'Reset trace scope to all traces',
                      description: 'Explicitly discard an invalid saved issue detection scope',
                    }),
                    disabled: scope.save.isLoading,
                    onClick: () => scope.save.mutate(emptyIssueDetectionTraceScope()),
                  },
                ]
              : []
          }
        />
      )}
      <Modal
        componentId={`${COMPONENT_ID}.modal`}
        visible={isOpen}
        title={intl.formatMessage({
          defaultMessage: 'Issue detection settings',
          description: 'Issue detection settings modal > Title',
        })}
        okText={intl.formatMessage({
          defaultMessage: 'Save',
          description: 'Issue detection settings modal > Save button',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Issue detection settings modal > Cancel button',
        })}
        confirmLoading={saveMutation.isLoading || scope.save.isLoading}
        okButtonProps={{ disabled: !canEdit || noFocusAreaSelected(draft) }}
        onOk={() => void save()}
        onCancel={close}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {(configurationError || loadError || saveMutation.error || scope.save.error) && (
            <Alert
              componentId={`${COMPONENT_ID}.error`}
              type="error"
              message={
                configurationError
                  ? configurationError.message
                  : scope.save.error instanceof Error
                    ? scope.save.error.message
                    : intl.formatMessage({
                        defaultMessage: 'Could not save issue detection settings. Try again.',
                        description: 'Issue detection settings modal > Save error',
                      })
              }
              closable={false}
            />
          )}
          <IssueDetectionSettingsForm
            key={openCount}
            draft={draft}
            setDraft={setDraft}
            disabled={!canEdit || saveMutation.isLoading || scope.save.isLoading}
          />
          <FormUI.Label>
            {intl.formatMessage({
              defaultMessage: 'Filter target traces',
              description: 'Issue detection settings trace scope label',
            })}
          </FormUI.Label>
          <IssueDetectionTraceScope
            scope={scopeDraft}
            onChange={setScopeDraft}
            disabled={!canEdit || saveMutation.isLoading || scope.save.isLoading}
          />
        </div>
      </Modal>
    </>
  );
};
