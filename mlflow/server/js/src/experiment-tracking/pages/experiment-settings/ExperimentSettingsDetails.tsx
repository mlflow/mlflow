import { Typography } from '@databricks/design-system';
import type { ReactNode } from 'react';
import { FormattedMessage } from 'react-intl';
import type { ExperimentEntity } from '../../types';
import { ExperimentViewArtifactLocation } from '../../components/experiment-page/components/ExperimentViewArtifactLocation';
import { ExperimentViewCopyArtifactLocation } from '../../components/experiment-page/components/header/ExperimentViewCopyArtifactLocation';
import { ExperimentViewCopyExperimentId } from '../../components/experiment-page/components/header/ExperimentViewCopyExperimentId';
import { ExperimentViewCopyTitle } from '../../components/experiment-page/components/header/ExperimentViewCopyTitle';
import { MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG } from '../../constants';
import { ExperimentSettingsNameInput } from './ExperimentSettingsNameInput';
import { ExperimentSettingsRow } from './ExperimentSettingsRow';

export interface ExperimentSettingsDetailsProps {
  experiment: ExperimentEntity;
  experimentKindSelector: ReactNode;
  canRename: boolean;
  saveExperimentName: (name: string) => Promise<void>;
}

export const ExperimentSettingsDetails = ({
  experiment,
  experimentKindSelector,
  canRename,
  saveExperimentName,
}: ExperimentSettingsDetailsProps) => {
  const experimentName = experiment.name.split('/').pop() ?? experiment.name;
  const traceLocation = experiment.tags?.find(
    (tag) => tag.key === MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG,
  )?.value;

  return (
    <>
      <section aria-labelledby="experiment-settings-about-heading">
        <div data-settings-section-content>
          <Typography.Title id="experiment-settings-about-heading" level={3} withoutMargins>
            <FormattedMessage defaultMessage="About" description="Heading for experiment metadata in Settings" />
          </Typography.Title>
          <ExperimentSettingsRow
            label={
              <FormattedMessage defaultMessage="Experiment name" description="Label for experiment name setting" />
            }
          >
            {(labelId) => (
              <ExperimentSettingsNameInput
                initialName={experimentName}
                isEditable={canRename}
                labelId={labelId}
                onSave={saveExperimentName}
              />
            )}
          </ExperimentSettingsRow>
          <ExperimentSettingsRow
            label={<FormattedMessage defaultMessage="Path" description="Label for experiment path setting" />}
            description={
              <FormattedMessage
                defaultMessage="The full path to this experiment."
                description="Description for experiment path setting"
              />
            }
          >
            <Typography.Text>{experiment.name}</Typography.Text>
            <ExperimentViewCopyTitle experiment={experiment} size="md" />
          </ExperimentSettingsRow>
          <ExperimentSettingsRow
            label={<FormattedMessage defaultMessage="Experiment ID" description="Label for experiment ID setting" />}
            description={
              <FormattedMessage
                defaultMessage="The unique identifier used by MLflow APIs."
                description="Description for experiment ID setting"
              />
            }
          >
            <Typography.Text>{experiment.experimentId}</Typography.Text>
            <ExperimentViewCopyExperimentId experiment={experiment} />
          </ExperimentSettingsRow>
          <ExperimentSettingsRow
            label={
              <FormattedMessage defaultMessage="Artifact location" description="Label for artifact location setting" />
            }
            description={
              <FormattedMessage
                defaultMessage="The location where run artifacts are stored."
                description="Description for artifact location setting"
              />
            }
          >
            <Typography.Text>
              <ExperimentViewArtifactLocation artifactLocation={experiment.artifactLocation} />
            </Typography.Text>
            <ExperimentViewCopyArtifactLocation experiment={experiment} />
          </ExperimentSettingsRow>
          <ExperimentSettingsRow
            label={
              <FormattedMessage defaultMessage="Experiment type" description="Label for experiment type setting" />
            }
            description={
              <FormattedMessage
                defaultMessage="Controls the tracking experience and available features."
                description="Description for experiment type setting"
              />
            }
          >
            {experimentKindSelector}
          </ExperimentSettingsRow>
        </div>
      </section>
      {traceLocation && (
        <section aria-labelledby="experiment-settings-trace-location-heading">
          <div data-settings-section-content>
            <Typography.Title id="experiment-settings-trace-location-heading" level={3} withoutMargins>
              <FormattedMessage defaultMessage="Trace location" description="Heading for experiment trace storage" />
            </Typography.Title>
            <ExperimentSettingsRow
              label={
                <FormattedMessage
                  defaultMessage="Trace storage destination"
                  description="Label for experiment trace storage destination"
                />
              }
              description={
                <FormattedMessage
                  defaultMessage="The destination configured for this experiment's traces."
                  description="Description for experiment trace storage destination"
                />
              }
            >
              <Typography.Text>{traceLocation}</Typography.Text>
            </ExperimentSettingsRow>
          </div>
        </section>
      )}
    </>
  );
};
