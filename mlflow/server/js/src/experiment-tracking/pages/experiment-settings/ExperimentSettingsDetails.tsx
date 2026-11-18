import { Button, CopyIcon, PencilIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { KeyValueTag } from '../../../common/components/KeyValueTag';
import { useTagAssignmentModal } from '../../../common/hooks/useTagAssignmentModal';
import type { KeyValueEntity } from '../../../common/types';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { isUserFacingTag } from '../../../common/utils/TagUtils';
import { CopyButton } from '../../../shared/building_blocks/CopyButton';
import { ExperimentViewArtifactLocation } from '../../components/experiment-page/components/ExperimentViewArtifactLocation';
import { EXPERIMENT_SETTINGS_TAGS_HASH } from '../../components/experiment-page/components/header/ExperimentViewManageTagsButton';
import { MlflowService } from '../../sdk/MlflowService';
import type { ExperimentEntity } from '../../types';
import { ExperimentSettingsNameInput } from './ExperimentSettingsNameInput';
import { ExperimentSettingsValue } from './ExperimentSettingsValue';
import { SettingCell } from './SettingCell';

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
  canEdit: boolean;
  refetchExperiment?: () => Promise<unknown>;
}

const ExperimentSettingsTags = ({ experiment, canEdit, refetchExperiment }: ExperimentSettingsTagsProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [isSavingTags, setIsSavingTags] = useState(false);
  const [saveTagsError, setSaveTagsError] = useState<string>();
  const hash = useLocation().hash;
  const editTagsRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (hash === EXPERIMENT_SETTINGS_TAGS_HASH) {
      editTagsRef.current?.scrollIntoView?.({ block: 'center' });
    }
  }, [hash]);
  const userFacingTags = (experiment.tags ?? []).filter((tag) => isUserFacingTag(tag.key));
  const { TagAssignmentModal, showTagAssignmentModal } = useTagAssignmentModal({
    componentIdPrefix: 'mlflow.experiment_settings.tags',
    initialTags: userFacingTags.map(({ key, value }) => ({ key, value })),
    isLoading: isSavingTags,
    error: saveTagsError,
    onClose: () => setSaveTagsError(undefined),
    disableSaveWithoutValidChanges: true,
    keepOpenOnSubmitError: true,
    onSubmit: (newTags: KeyValueEntity[], deletedTags: KeyValueEntity[]) => {
      const experimentId = experiment.experimentId;
      setIsSavingTags(true);
      setSaveTagsError(undefined);
      return Promise.all([
        ...newTags.map(({ key, value }) => MlflowService.setExperimentTag({ experiment_id: experimentId, key, value })),
        ...deletedTags.map(({ key }) => MlflowService.deleteExperimentTag({ experiment_id: experimentId, key })),
      ])
        .catch((error) => {
          setSaveTagsError(error instanceof ErrorWrapper ? error.getMessageField() : error?.message);
          // Rethrow so the modal stays open and shows the error.
          throw error;
        })
        .finally(() => refetchExperiment?.())
        .finally(() => setIsSavingTags(false));
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
            <Tooltip
              componentId="mlflow.experiment_settings.tags.edit_tooltip"
              content={
                canEdit
                  ? editTagsLabel
                  : intl.formatMessage({
                      defaultMessage: 'You do not have permission to edit tags on this experiment.',
                      description: 'Tooltip shown when the user cannot edit experiment tags in Settings',
                    })
              }
            >
              <span ref={editTagsRef}>
                <Button
                  componentId="mlflow.experiment_settings.tags.edit"
                  type="tertiary"
                  size="small"
                  icon={<PencilIcon />}
                  aria-label={editTagsLabel}
                  onClick={canEdit ? showTagAssignmentModal : undefined}
                  disabled={!canEdit}
                />
              </span>
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
  experimentKindSelector: ReactNode;
  canEdit: boolean;
  saveExperimentName: (name: string) => Promise<void>;
  refetchExperiment?: () => Promise<unknown>;
}

/** The About section: name, ID, artifact location, experiment type and tags. */
export const ExperimentSettingsDetails = ({
  experiment,
  experimentKindSelector,
  canEdit,
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
                isEditable={canEdit}
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
          {() => (
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
          )}
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
          {() => (
            <ExperimentSettingsValue>
              <Typography.Text>
                <ExperimentViewArtifactLocation artifactLocation={experiment.artifactLocation} />
              </Typography.Text>
              <CopyValueButton
                componentId="mlflow.experiment_settings.copy_artifact_location"
                copyText={experiment.artifactLocation}
                label={intl.formatMessage({
                  defaultMessage: 'Copy artifact location',
                  description:
                    'Accessible label for the button that copies the experiment artifact location in Settings',
                })}
              />
            </ExperimentSettingsValue>
          )}
        </SettingCell.Static>
        <SettingCell
          label={
            <>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Experiment type"
                  description="Label for the experiment type setting"
                />
              </Typography.Text>
              <Typography.Hint>
                <FormattedMessage
                  defaultMessage="Controls the tracking experience and available features."
                  description="Description for the experiment type setting"
                />
              </Typography.Hint>
            </>
          }
          value={<ExperimentSettingsValue>{experimentKindSelector}</ExperimentSettingsValue>}
        />
        <ExperimentSettingsTags experiment={experiment} canEdit={canEdit} refetchExperiment={refetchExperiment} />
      </div>
    </section>
  );
};
