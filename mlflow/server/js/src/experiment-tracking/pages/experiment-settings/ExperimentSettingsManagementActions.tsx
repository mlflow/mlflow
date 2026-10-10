import { Button, Typography } from '@databricks/design-system';
import type { ReactNode } from 'react';
import { FormattedMessage } from 'react-intl';
import type { ExperimentEntity } from '../../types';
import { useExperimentManagementActions } from '../../components/experiment-page/hooks/useExperimentManagementActions';
import { ExperimentSettingsDetails } from './ExperimentSettingsDetails';
import { ExperimentSettingsRow } from './ExperimentSettingsRow';

export interface ExperimentSettingsContentProps {
  experiment: ExperimentEntity;
  experimentKindSelector: ReactNode;
  refetchExperiment?: () => Promise<unknown>;
}

export const ExperimentSettingsContent = ({
  experiment,
  experimentKindSelector,
  refetchExperiment,
}: ExperimentSettingsContentProps) => {
  const actions = useExperimentManagementActions({ experiment, refetchExperiment });

  return (
    <>
      <ExperimentSettingsDetails
        experiment={experiment}
        experimentKindSelector={experimentKindSelector}
        canRename={actions.canRename}
        saveExperimentName={actions.saveExperimentName}
      />
      <section aria-labelledby="experiment-settings-actions-heading">
        <div data-settings-section-content>
          <Typography.Title id="experiment-settings-actions-heading" level={3} withoutMargins>
            <FormattedMessage defaultMessage="Actions" description="Heading for experiment management actions" />
          </Typography.Title>
          {actions.canDelete && (
            <ExperimentSettingsRow
              label={
                <FormattedMessage defaultMessage="Delete experiment" description="Label for deleting experiment" />
              }
              description={
                <FormattedMessage
                  defaultMessage="Moves this experiment to the trash."
                  description="Description for deleting experiment"
                />
              }
            >
              <Button componentId="mlflow.experiment_settings.delete" onClick={actions.deleteExperiment} danger>
                <FormattedMessage defaultMessage="Delete" description="Button to delete experiment" />
              </Button>
            </ExperimentSettingsRow>
          )}
        </div>
      </section>
      {actions.overlayBodies}
    </>
  );
};
