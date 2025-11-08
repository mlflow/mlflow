import React from 'react';
import { useCallback, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import {
  discardPendingEvaluationData,
  writeBackEvaluationArtifactsAction,
} from '../../../actions/PromptEngineeringActions';
import { FormattedMessage } from 'react-intl';
import Utils from '../../../../common/utils/Utils';
import { useBrowserKeyShortcutListener } from '../../../../common/hooks/useBrowserKeyShortcutListener';

export const useEvaluationArtifactWriteBack = () => {
  const { evaluationPendingDataByRunUuid, evaluationArtifactsBeingUploaded, evaluationDraftInputValues } = useSelector(
    ({ evaluationData }: ReduxState) => evaluationData,
  );

  const [isSyncingArtifacts, setSyncingArtifacts] = useState(false);

  const dispatch = useDispatch<ThunkDispatch>();

  const discard = useCallback(() => {
    dispatch(discardPendingEvaluationData());
  }, [dispatch]);

  const unsyncedDataEntriesCount = Object.values(evaluationPendingDataByRunUuid).flat().length;
  const draftInputValuesCount = evaluationDraftInputValues.length;
  const runsBeingSynchronizedCount = Object.values(evaluationArtifactsBeingUploaded).filter((runArtifacts) =>
    Object.values(runArtifacts).some((isSynced) => isSynced),
  ).length;

  useEffect(() => {
    if (unsyncedDataEntriesCount === 0) {
      setSyncingArtifacts(false);
    }
  }, [unsyncedDataEntriesCount]);

  const synchronizeArtifactData = useCallback(() => {
    if (unsyncedDataEntriesCount === 0 || isSyncingArtifacts) {
      return true;
    }
    setSyncingArtifacts(true);
    dispatch(writeBackEvaluationArtifactsAction()).catch((e) => {
      Utils.logErrorAndNotifyUser(e);
    });
    return true;
  }, [dispatch, unsyncedDataEntriesCount, isSyncingArtifacts]);

  const { isMacKeyboard } = useBrowserKeyShortcutListener('s', { ctrlOrCmdKey: true }, synchronizeArtifactData);

  const { theme } = useDesignSystemTheme();

  // Following flag is true when there are draft input values (draft rows), but
  // no evaluated values yet
  const pendingUnevaluatedDraftInputValues = draftInputValuesCount > 0 && unsyncedDataEntriesCount === 0;

  // Display write-back UI only if there are draft rows or unsynced evaluation values
  const shouldStatusElementBeDisplayed = unsyncedDataEntriesCount > 0 || pendingUnevaluatedDraftInputValues;

  const EvaluationSyncStatusElement = shouldStatusElementBeDisplayed ? (
    <div
      css={{
        backgroundColor: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.border}`,
        padding: theme.spacing.md,
        marginBottom: theme.spacing.sm,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}
    >
      {pendingUnevaluatedDraftInputValues ? (
        <FormattedMessage
          defaultMessage="You have added rows with new input values, but you still need to evaluate the new data in order to save it."
          description="Experiment page > artifact compare view > prompt lab artifact synchronization > unevaluated rows indicator"
        />
      ) : isSyncingArtifacts ? (
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Synchronizing artifacts for {runsBeingSynchronizedCount} runs..."
            description="Experiment page > artifact compare view > prompt lab artifact synchronization > loading state"
            values={{
              runsBeingSynchronizedCount: <strong>{runsBeingSynchronizedCount}</strong>,
            }}
          />
        </Typography.Text>
      ) : (
        <Typography.Text>
          <FormattedMessage
            defaultMessage={`You have <strong>{unsyncedDataEntriesCount}</strong> unsaved evaluated {unsyncedDataEntriesCount, plural, =1 {value} other {values}}. Click "Save" button or press {keyCombination} keys to synchronize the artifact data.`}
            description="Experiment page > artifact compare view > prompt lab artifact synchronization > pending changes indicator"
            values={{
              strong: (value) => <strong>{value}</strong>,
              unsyncedDataEntriesCount,
              keyCombination: isMacKeyboard() ? '⌘CMD+S' : 'CTRL+S',
            }}
          />
        </Typography.Text>
      )}
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationartifactwriteback.tsx_102"
          disabled={isSyncingArtifacts}
          onClick={discard}
        >
          <FormattedMessage
            defaultMessage="Discard"
            description="Experiment page > artifact compare view > prompt lab artifact synchronization > submit button label"
          />
        </Button>{' '}
        {/* Display "Save" button only if there are actual evaluated data to sync (don't allow to sync empty draft rows) */}
        {unsyncedDataEntriesCount > 0 && (
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationartifactwriteback.tsx_110"
            loading={isSyncingArtifacts}
            type="primary"
            onClick={synchronizeArtifactData}
          >
            <FormattedMessage
              defaultMessage="Save"
              description="Experiment page > artifact compare view > prompt lab artifact synchronization > cancel button label"
            />
          </Button>
        )}
      </div>
    </div>
  ) : null;

  return { isSyncingArtifacts, EvaluationSyncStatusElement };
};
