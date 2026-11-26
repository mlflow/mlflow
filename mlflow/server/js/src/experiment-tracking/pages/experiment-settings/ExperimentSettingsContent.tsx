import { Button, Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { ExperimentEntity } from '../../types';
import { useExperimentManagementActions } from '../../components/experiment-page/hooks/useExperimentManagementActions';
import { ExperimentSettingsDetails } from './ExperimentSettingsDetails';
import { ExperimentSettingsValue } from './ExperimentSettingsValue';
import { SettingCell } from './SettingCell';

export interface ExperimentSettingsContentProps {
  experiment: ExperimentEntity;
  refetchExperiment?: () => Promise<unknown>;
}

export const ExperimentSettingsContent = ({ experiment, refetchExperiment }: ExperimentSettingsContentProps) => {
  const actions = useExperimentManagementActions({ experiment, refetchExperiment });

  return (
    <>
      <ExperimentSettingsDetails
        experiment={experiment}
        saveExperimentName={actions.saveExperimentName}
        refetchExperiment={refetchExperiment}
      />
      <section aria-labelledby="experiment-settings-actions-heading">
        <div data-settings-section-content>
          <Typography.Title id="experiment-settings-actions-heading" level={3} withoutMargins>
            <FormattedMessage defaultMessage="Actions" description="Heading for other experiment actions" />
          </Typography.Title>
          <SettingCell
            label={
              <>
                <Typography.Text bold>
                  <FormattedMessage
                    defaultMessage="Delete experiment"
                    description="Label for deleting the experiment"
                  />
                </Typography.Text>
                <Typography.Hint>
                  <FormattedMessage
                    defaultMessage="Deletes this experiment. Deleted experiments can be restored."
                    description="Description for deleting the experiment in OSS MLflow, where deletion is a soft delete"
                  />
                </Typography.Hint>
              </>
            }
            value={
              <ExperimentSettingsValue>
                <Button componentId="mlflow.experiment_settings.delete" onClick={actions.deleteExperiment} danger>
                  <FormattedMessage defaultMessage="Delete" description="Button to delete the experiment" />
                </Button>
              </ExperimentSettingsValue>
            }
          />
        </div>
      </section>
      {actions.overlayBodies}
    </>
  );
};
