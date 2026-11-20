import { Button, Tooltip, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCurrentUserAdminWorkspaces, useCurrentUserIsAdmin, useCurrentUserQuery } from '../../../account/hooks';
import AdminRoutes from '../../../admin/routes';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import type { ExperimentEntity } from '../../types';
import { useWorkspacesEnabled } from '../../hooks/useServerInfo';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { useExperimentManagementActions } from '../../components/experiment-page/hooks/useExperimentManagementActions';
import { ExperimentSettingsDetails } from './ExperimentSettingsDetails';
import { ExperimentSettingsValue } from './ExperimentSettingsValue';
import { SettingCell } from './SettingCell';

/** Shown only when the server has authentication enabled; permissions are managed on the admin page. */
const ExperimentSettingsPermissions = () => {
  const intl = useIntl();
  const navigate = useNavigate();
  const { data: currentUser } = useCurrentUserQuery();
  const isAdmin = useCurrentUserIsAdmin();
  const adminWorkspaces = useCurrentUserAdminWorkspaces();
  const activeWorkspace = useActiveWorkspace();
  const { workspacesEnabled, loading: workspacesLoading } = useWorkspacesEnabled();
  // Only a manager of the workspace being viewed, like the edit and delete checks. Without workspaces
  // there is a single workspace, so managing any workspace counts.
  const workspacesDefinitelyDisabled = !workspacesEnabled && !workspacesLoading;
  const isWorkspaceAdmin = workspacesDefinitelyDisabled
    ? adminWorkspaces.size > 0
    : Boolean(activeWorkspace && adminWorkspaces.has(activeWorkspace));

  if (!currentUser?.user) {
    return null;
  }

  const canManagePermissions = isAdmin || isWorkspaceAdmin;

  return (
    <section aria-labelledby="experiment-settings-permissions-heading">
      <div data-settings-section-content>
        <Typography.Title id="experiment-settings-permissions-heading" level={3} withoutMargins>
          <FormattedMessage defaultMessage="Permissions" description="Heading for experiment permission settings" />
        </Typography.Title>
        <SettingCell
          label={
            <>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Experiment permissions"
                  description="Label for the experiment permissions setting"
                />
              </Typography.Text>
              <Typography.Hint>
                <FormattedMessage
                  defaultMessage="Controls who can view and manage this experiment. Permissions are granted to users and roles by an admin."
                  description="Description for the experiment permissions setting on an MLflow server with authentication"
                />
              </Typography.Hint>
            </>
          }
          value={
            <ExperimentSettingsValue>
              <Tooltip
                componentId="mlflow.experiment_settings.permissions.tooltip"
                content={
                  canManagePermissions
                    ? undefined
                    : intl.formatMessage({
                        defaultMessage: 'Only admins can manage permissions. Ask an admin for access.',
                        description: 'Tooltip shown when the user cannot manage experiment permissions in Settings',
                      })
                }
              >
                <span>
                  <Button
                    componentId="mlflow.experiment_settings.permissions"
                    onClick={canManagePermissions ? () => navigate(AdminRoutes.adminPageRoute) : undefined}
                    disabled={!canManagePermissions}
                  >
                    <FormattedMessage
                      defaultMessage="Manage permissions"
                      description="Button that opens the admin page where experiment permissions are managed"
                    />
                  </Button>
                </span>
              </Tooltip>
            </ExperimentSettingsValue>
          }
        />
      </div>
    </section>
  );
};

export interface ExperimentSettingsContentProps {
  experiment: ExperimentEntity;
  refetchExperiment?: () => Promise<unknown>;
}

export const ExperimentSettingsContent = ({ experiment, refetchExperiment }: ExperimentSettingsContentProps) => {
  const intl = useIntl();
  const actions = useExperimentManagementActions({ experiment, refetchExperiment });

  return (
    <>
      <ExperimentSettingsDetails
        experiment={experiment}
        canEdit={actions.canEdit}
        saveExperimentName={actions.saveExperimentName}
        refetchExperiment={refetchExperiment}
      />
      <ExperimentSettingsPermissions />
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
                <Tooltip
                  componentId="mlflow.experiment_settings.delete.tooltip"
                  content={
                    actions.canDelete
                      ? undefined
                      : intl.formatMessage({
                          defaultMessage: 'You need Manage permission on this experiment to delete it.',
                          description: 'Tooltip shown when the user cannot delete the experiment in Settings',
                        })
                  }
                >
                  <span>
                    <Button
                      componentId="mlflow.experiment_settings.delete"
                      onClick={actions.canDelete ? actions.deleteExperiment : undefined}
                      disabled={!actions.canDelete}
                      danger
                    >
                      <FormattedMessage defaultMessage="Delete" description="Button to delete the experiment" />
                    </Button>
                  </span>
                </Tooltip>
              </ExperimentSettingsValue>
            }
          />
        </div>
      </section>
      {actions.overlayBodies}
    </>
  );
};
