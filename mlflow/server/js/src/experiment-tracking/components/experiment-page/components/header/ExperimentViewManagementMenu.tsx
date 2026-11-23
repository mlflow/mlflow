import React, { useState } from 'react';
import { Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { OverflowMenu } from '../../../../../shared/building_blocks/PageHeader';
import type { ExperimentEntity } from '../../../../types';
import { useLocation, useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { getPreservedQueryString } from '../../../../pages/experiment-page-tabs/side-nav/utils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { DeleteExperimentModal } from '../../../modals/DeleteExperimentModal';
import { useInvalidateExperimentList } from '../../hooks/useExperimentListQuery';
import { canDeleteExperiment, canModifyExperiment, canRenameExperiment } from '../../utils/experimentPage.common-utils';
import { shouldEnableExperimentSettingsPage } from '../../../../../common/utils/FeatureUtils';
import { ExperimentPageTabName } from '../../../../constants';

/**
 * Experiment page header part responsible for displaying menu
 * with edit and delete buttons
 */
export const ExperimentViewManagementMenu = ({
  experiment,
  setEditing,
  baseComponentId = 'mlflow.experiment_page.managementMenu',
}: {
  experiment: ExperimentEntity;
  setEditing?: (editing: boolean) => void;
  baseComponentId?: string;
}) => {
  const [showDeleteExperimentModal, setShowDeleteExperimentModal] = useState(false);
  const invalidateExperimentList = useInvalidateExperimentList();
  const navigate = useNavigate();
  const { search } = useLocation();
  // With the Settings page, the name and trace archival retention are edited there, so the editor only
  // edits the description.
  const settingsPageEnabled = shouldEnableExperimentSettingsPage();
  const canEditExperiment = Boolean(setEditing) && (canRenameExperiment(experiment) || canModifyExperiment(experiment));
  const canEditDescription = Boolean(setEditing) && canModifyExperiment(experiment);
  const canDelete = canDeleteExperiment(experiment);
  const menu = [
    ...(settingsPageEnabled
      ? [
          {
            id: `${baseComponentId}.settings`,
            itemName: (
              <Typography.Text>
                <FormattedMessage defaultMessage="Settings" description="Link to the experiment settings page" />
              </Typography.Text>
            ),
            onClick: () =>
              navigate(
                `${Routes.getExperimentPageTabRoute(experiment.experimentId, ExperimentPageTabName.Settings)}${
                  getPreservedQueryString(search) ?? ''
                }`,
              ),
          },
        ]
      : []),
    ...(settingsPageEnabled && canEditDescription
      ? [
          {
            id: `${baseComponentId}.edit-description`,
            itemName: (
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="Edit description"
                  description="Text for the menu item that edits the experiment description on the experiment view page header"
                />
              </Typography.Text>
            ),
            onClick: () => setEditing?.(true),
          },
        ]
      : []),
    ...(!settingsPageEnabled && canEditExperiment
      ? [
          {
            id: `${baseComponentId}.edit-experiment`,
            itemName: (
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="Edit experiment"
                  description="Text for edit experiment button on experiment view page header"
                />
              </Typography.Text>
            ),
            onClick: () => setEditing?.(true),
          },
        ]
      : []),
    ...(canDelete
      ? [
          {
            id: `${baseComponentId}.delete`,
            itemName: (
              <FormattedMessage
                defaultMessage="Delete"
                description="Text for delete button on the experiment view page header"
              />
            ),
            onClick: () => setShowDeleteExperimentModal(true),
          },
        ]
      : []),
  ];

  if (menu.length === 0) {
    return null;
  }

  return (
    <>
      <OverflowMenu menu={menu} />
      <DeleteExperimentModal
        experimentId={experiment.experimentId}
        experimentName={experiment.name}
        isOpen={showDeleteExperimentModal}
        onClose={() => setShowDeleteExperimentModal(false)}
        onExperimentDeleted={() => {
          invalidateExperimentList();
          navigate(Routes.experimentsObservatoryRoute);
        }}
      />
    </>
  );
};
