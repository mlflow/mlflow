import { Button, CopyIcon, PencilIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { KeyValueTag } from '../../../common/components/KeyValueTag';
import { useTagAssignmentModal } from '../../../common/hooks/useTagAssignmentModal';
import type { KeyValueEntity } from '../../../common/types';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { isUserFacingTag } from '../../../common/utils/TagUtils';
import { CopyButton } from '../../../shared/building_blocks/CopyButton';
import { ExperimentViewArtifactLocation } from '../../components/experiment-page/components/ExperimentViewArtifactLocation';
import { MlflowService } from '../../sdk/MlflowService';
import type { ExperimentEntity } from '../../types';
import { ExperimentSettingsNameInput } from './ExperimentSettingsNameInput';
import { ExperimentSettingsTraceArchivalRetention } from './ExperimentSettingsTraceArchivalRetention';
import { ExperimentSettingsValue } from './ExperimentSettingsValue';
import { SettingCell } from './SettingCell';
import { getPermissionDeniedReason } from './useSettingAutosave';

const CopyValueButton = ({
  componentId,
  copyText,
  label,
}: {
  componentId: string;
  copyText: string;
  label: string;
}) => (
  <CopyButton
    componentId={componentId}
    copyText={copyText}
    showLabel={false}
    type="tertiary"
    size="small"
    icon={<CopyIcon />}
    aria-label={label}
  />
);

interface ExperimentSettingsTagsProps {
  experiment: ExperimentEntity;
  refetchExperiment?: () => Promise<unknown>;
}

const ExperimentSettingsTags = ({ experiment, refetchExperiment }: ExperimentSettingsTagsProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [isSavingTags, setIsSavingTags] = useState(false);
  const [saveTagsError, setSaveTagsError] = useState<string>();
  // The tags the server has after a partly failed save. The experiment isn't refetched then (that
  // would reset the editor and drop the failed edits), so the editor keeps diffing against the tags
  // it opened with, and the retry diffs its result against these instead.
  const serverTagsAfterPartialSave = useRef<Map<string, string> | null>(null);
  const userFacingTags = (experiment.tags ?? []).filter((tag) => isUserFacingTag(tag.key));
  const { TagAssignmentModal, showTagAssignmentModal } = useTagAssignmentModal({
    componentIdPrefix: 'mlflow.experiment_settings.tags',
    initialTags: userFacingTags.map(({ key, value }) => ({ key, value })),
    isLoading: isSavingTags,
    error: saveTagsError,
    onClose: () => {
      setSaveTagsError(undefined);
      if (serverTagsAfterPartialSave.current) {
        serverTagsAfterPartialSave.current = null;
        refetchExperiment?.()?.catch(() => undefined);
      }
    },
    disableSaveWithoutValidChanges: true,
    catchSubmitError: true,
    onSubmit: async (newTags: KeyValueEntity[], deletedTags: KeyValueEntity[]) => {
      const experimentId = experiment.experimentId;
      const openedWith = new Map(userFacingTags.map(({ key, value }) => [key, value]));
      const serverTags = serverTagsAfterPartialSave.current ?? new Map(openedWith);
      // The editor reports changes against the tags it opened with; rebuild the tags it now shows.
      const wantedTags = new Map(openedWith);
      deletedTags.forEach(({ key }) => wantedTags.delete(key));
      newTags.forEach(({ key, value }) => wantedTags.set(key, value));
      const changes = [
        ...[...wantedTags]
          .filter(([key, value]) => serverTags.get(key) !== value)
          .map(([key, value]) => ({
            save: () => MlflowService.setExperimentTag({ experiment_id: experimentId, key, value }),
            applyToServerTags: () => serverTags.set(key, value),
          })),
        ...[...serverTags.keys()]
          .filter((key) => !wantedTags.has(key))
          .map((key) => ({
            save: () => MlflowService.deleteExperimentTag({ experiment_id: experimentId, key }),
            applyToServerTags: () => serverTags.delete(key),
          })),
      ];
      setIsSavingTags(true);
      setSaveTagsError(undefined);
      const results = await Promise.allSettled(changes.map((change) => change.save()));
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          changes[index].applyToServerTags();
        }
      });
      setIsSavingTags(false);
      const failure = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (failure) {
        serverTagsAfterPartialSave.current = serverTags;
        const error = failure.reason;
        setSaveTagsError(
          getPermissionDeniedReason(error) ??
            (error instanceof ErrorWrapper ? error.getMessageField() : error?.message),
        );
        // Rejecting keeps the editor open with the error and the user's edits.
        throw error;
      }
      serverTagsAfterPartialSave.current = null;
      // The tags have saved; a failed refresh afterwards isn't a failed save.
      await refetchExperiment?.()?.catch(() => undefined);
    },
  });
  const editTagsLabel = intl.formatMessage({
    defaultMessage: 'Edit tags',
    description: 'Accessible label and tooltip for the icon button that edits experiment tags in Settings',
  });

  return (
    <SettingCell
      label={
        <>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Tags" description="Label for the experiment tags setting" />
          </Typography.Text>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Key-value labels for organizing and finding this experiment."
              description="Description for the experiment tags setting"
            />
          </Typography.Hint>
        </>
      }
      value={
        <ExperimentSettingsValue>
          <div css={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: theme.spacing.xs }}>
            {userFacingTags.map((tag) => (
              <KeyValueTag key={tag.key} tag={tag} enableFullViewModal />
            ))}
            <Tooltip componentId="mlflow.experiment_settings.tags.edit_tooltip" content={editTagsLabel}>
              <Button
                componentId="mlflow.experiment_settings.tags.edit"
                type="tertiary"
                size="small"
                icon={<PencilIcon />}
                aria-label={editTagsLabel}
                onClick={showTagAssignmentModal}
              />
            </Tooltip>
          </div>
          {TagAssignmentModal}
        </ExperimentSettingsValue>
      }
    />
  );
};

export interface ExperimentSettingsDetailsProps {
  experiment: ExperimentEntity;
  saveExperimentName: (name: string) => Promise<void>;
  refetchExperiment?: () => Promise<unknown>;
}

/** The About section: name, ID, artifact location, tags and, with trace archival on, its retention. */
export const ExperimentSettingsDetails = ({
  experiment,
  saveExperimentName,
  refetchExperiment,
}: ExperimentSettingsDetailsProps) => {
  const intl = useIntl();

  return (
    <section aria-labelledby="experiment-settings-about-heading">
      <div data-settings-section-content>
        <Typography.Title id="experiment-settings-about-heading" level={3} withoutMargins>
          <FormattedMessage defaultMessage="About" description="Heading for experiment metadata in Settings" />
        </Typography.Title>
        <SettingCell.Control
          label={
            <FormattedMessage defaultMessage="Experiment name" description="Label for the experiment name setting" />
          }
        >
          {({ labelId }) => (
            <ExperimentSettingsValue>
              <ExperimentSettingsNameInput
                initialName={experiment.name}
                labelId={labelId}
                onSave={saveExperimentName}
              />
            </ExperimentSettingsValue>
          )}
        </SettingCell.Control>
        <SettingCell.Static
          label={<FormattedMessage defaultMessage="Experiment ID" description="Label for the experiment ID setting" />}
          description={
            <FormattedMessage
              defaultMessage="The unique identifier used by MLflow APIs."
              description="Description for the experiment ID setting"
            />
          }
        >
          <ExperimentSettingsValue>
            <Typography.Text>{experiment.experimentId}</Typography.Text>
            <CopyValueButton
              componentId="mlflow.experiment_settings.copy_experiment_id"
              copyText={experiment.experimentId}
              label={intl.formatMessage({
                defaultMessage: 'Copy experiment ID',
                description: 'Accessible label for the button that copies the experiment ID in Settings',
              })}
            />
          </ExperimentSettingsValue>
        </SettingCell.Static>
        <SettingCell.Static
          label={
            <FormattedMessage
              defaultMessage="Artifact location"
              description="Label for the experiment artifact location setting"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="The location where run artifacts are stored."
              description="Description for the experiment artifact location setting"
            />
          }
        >
          <ExperimentSettingsValue>
            <Typography.Text>
              <ExperimentViewArtifactLocation artifactLocation={experiment.artifactLocation} />
            </Typography.Text>
            <CopyValueButton
              componentId="mlflow.experiment_settings.copy_artifact_location"
              copyText={experiment.artifactLocation}
              label={intl.formatMessage({
                defaultMessage: 'Copy artifact location',
                description: 'Accessible label for the button that copies the experiment artifact location in Settings',
              })}
            />
          </ExperimentSettingsValue>
        </SettingCell.Static>
        <ExperimentSettingsTags experiment={experiment} refetchExperiment={refetchExperiment} />
        <ExperimentSettingsTraceArchivalRetention experiment={experiment} refetchExperiment={refetchExperiment} />
      </div>
    </section>
  );
};
