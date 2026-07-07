import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  GearIcon,
  Modal,
  Pagination,
  PencilIcon,
  Spacer,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormProvider, useForm } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../common/utils/RoutingUtils';
import { useCurrentUserAdminWorkspaces, useCurrentUserIsAdmin, useIsAuthAvailable } from '../../account/hooks';
import {
  DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT,
  formatTraceArchivalRetention,
  getTraceArchivalRetentionValidationError,
  parseTraceArchivalRetention,
  type TraceArchivalRetentionUnit,
} from '../../common/utils/traceArchival';
import { useWorkspaces, type Workspace } from '../../workspaces/hooks/useWorkspaces';
import {
  getLastUsedWorkspace,
  setLastUsedWorkspace,
  WORKSPACE_QUERY_PARAM,
} from '../../workspaces/utils/WorkspaceUtils';
import { useUpdateWorkspace } from '../../workspaces/hooks/useUpdateWorkspace';
import { WorkspaceSettingsFields } from './WorkspaceSettingsFields';
import { useTraceArchivalEnabled } from '../../experiment-tracking/hooks/useServerInfo';

type WorkspacesHomeViewProps = {
  onCreateWorkspace: () => void;
};

type EditWorkspaceFormData = {
  description: string;
  defaultArtifactRoot: string;
  traceArchivalLocation: string;
};

const WorkspacesEmptyState = ({ onCreateWorkspace }: { onCreateWorkspace: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        padding: theme.spacing.lg,
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      <Typography.Title level={4} css={{ margin: 0 }}>
        <FormattedMessage
          defaultMessage="Create your first workspace"
          description="Home page workspaces empty state title"
        />
      </Typography.Title>
      <Typography.Text css={{ color: theme.colors.textSecondary }}>
        <FormattedMessage
          defaultMessage="Create a workspace to organize and logically isolate your experiments and models."
          description="Home page workspaces empty state description"
        />
      </Typography.Text>
      <Button componentId="mlflow.home.workspaces.create" onClick={onCreateWorkspace}>
        <FormattedMessage defaultMessage="Create workspace" description="Home page workspaces empty state CTA" />
      </Button>
    </div>
  );
};

const WorkspaceRow = ({
  workspace,
  isLastUsed,
  canEdit,
  canManage,
  showEditColumn,
  showManageColumn,
  traceArchivalEnabled,
}: {
  workspace: Workspace;
  isLastUsed: boolean;
  canEdit: boolean;
  canManage: boolean;
  showEditColumn: boolean;
  showManageColumn: boolean;
  traceArchivalEnabled: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { mutate: updateWorkspace, isLoading: isPending } = useUpdateWorkspace();
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [traceArchivalRetentionAmount, setTraceArchivalRetentionAmount] = useState('');
  const [traceArchivalRetentionUnit, setTraceArchivalRetentionUnit] = useState<TraceArchivalRetentionUnit>(
    DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT,
  );
  const [traceArchivalRetentionError, setTraceArchivalRetentionError] = useState<string | undefined>();
  const [initialTraceArchivalRetention, setInitialTraceArchivalRetention] = useState('');
  const [didEditTraceArchivalRetention, setDidEditTraceArchivalRetention] = useState(false);
  const form = useForm<EditWorkspaceFormData>({
    defaultValues: {
      description: '',
      defaultArtifactRoot: '',
      traceArchivalLocation: '',
    },
  });

  const updateTraceArchivalRetention = ({
    amount = traceArchivalRetentionAmount,
    unit = traceArchivalRetentionUnit,
  }: {
    amount?: string;
    unit?: TraceArchivalRetentionUnit;
  }) => {
    setTraceArchivalRetentionAmount(amount);
    setTraceArchivalRetentionUnit(unit);
    setTraceArchivalRetentionError(getTraceArchivalRetentionValidationError(amount, unit, intl));
  };

  const handleNameClick = () => {
    // Persist to localStorage for UI hints then hard reload to cleanly switch workspace
    setLastUsedWorkspace(workspace.name);
    window.location.hash = `#/?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(workspace.name)}`;
    window.location.reload();
  };

  const handleManageClick = () => {
    // Same hard-reload pattern as handleNameClick — flushes any stale auth /
    // workspace context before landing on the per-workspace management view.
    // ``/admin/ws`` is the workspace-mode pathname; the workspace value still
    // travels in ``?workspace=`` so ``WorkspaceRouterSync`` keeps the global
    // ``activeWorkspace`` in sync.
    setLastUsedWorkspace(workspace.name);
    window.location.hash = `#/admin/ws?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(workspace.name)}`;
    window.location.reload();
  };

  const handleEditClick = () => {
    setEditError(null);
    form.reset({
      description: workspace.description ?? '',
      defaultArtifactRoot: workspace.default_artifact_root ?? '',
      traceArchivalLocation: workspace.trace_archival_config?.location ?? '',
    });
    const rawTraceArchivalRetention = workspace.trace_archival_config?.retention?.trim() ?? '';
    const { amount, unit } = parseTraceArchivalRetention(rawTraceArchivalRetention);
    setInitialTraceArchivalRetention(rawTraceArchivalRetention);
    setDidEditTraceArchivalRetention(false);
    updateTraceArchivalRetention({ amount, unit });
    setIsEditModalVisible(true);
  };

  const handleSave = form.handleSubmit((values) => {
    const currentDescription = workspace.description ?? '';
    const currentDefaultArtifactRoot = workspace.default_artifact_root ?? '';
    const currentTraceArchivalLocation = workspace.trace_archival_config?.location?.trim() ?? '';
    const currentTraceArchivalRetention = initialTraceArchivalRetention;
    const trimmedTraceArchivalLocation = values.traceArchivalLocation.trim();
    const trimmedTraceArchivalRetention = didEditTraceArchivalRetention
      ? formatTraceArchivalRetention(traceArchivalRetentionAmount, traceArchivalRetentionUnit)
      : currentTraceArchivalRetention;
    const nextTraceArchivalRetentionError = didEditTraceArchivalRetention
      ? getTraceArchivalRetentionValidationError(traceArchivalRetentionAmount, traceArchivalRetentionUnit, intl)
      : undefined;
    setTraceArchivalRetentionError(nextTraceArchivalRetentionError);
    if (nextTraceArchivalRetentionError) {
      return;
    }

    const hasDescriptionChanged = values.description !== currentDescription;
    const hasDefaultArtifactRootChanged = values.defaultArtifactRoot !== currentDefaultArtifactRoot;
    const hasTraceArchivalLocationChanged = trimmedTraceArchivalLocation !== currentTraceArchivalLocation;
    const hasTraceArchivalRetentionChanged = trimmedTraceArchivalRetention !== currentTraceArchivalRetention;

    if (
      !hasDescriptionChanged &&
      !hasDefaultArtifactRootChanged &&
      !hasTraceArchivalLocationChanged &&
      !hasTraceArchivalRetentionChanged
    ) {
      setIsEditModalVisible(false);
      setEditError(null);
      return;
    }

    setEditError(null);
    updateWorkspace(
      {
        name: workspace.name,
        ...(hasDescriptionChanged ? { description: values.description } : {}),
        ...(hasDefaultArtifactRootChanged ? { default_artifact_root: values.defaultArtifactRoot } : {}),
        ...(hasTraceArchivalLocationChanged || hasTraceArchivalRetentionChanged
          ? {
              trace_archival_config: {
                ...(hasTraceArchivalLocationChanged ? { location: trimmedTraceArchivalLocation } : {}),
                ...(hasTraceArchivalRetentionChanged ? { retention: trimmedTraceArchivalRetention } : {}),
              },
            }
          : {}),
      },
      {
        onSuccess: () => {
          setIsEditModalVisible(false);
          setEditError(null);
        },
        onError: (error: any) => {
          setEditError(
            error?.message ||
              intl.formatMessage({
                defaultMessage: 'Failed to update workspace. Please try again.',
                description: 'Generic error message for edit workspace modal',
              }),
          );
        },
      },
    );
  });

  const handleCancel = () => {
    setEditError(null);
    setTraceArchivalRetentionError(undefined);
    setDidEditTraceArchivalRetention(false);
    setIsEditModalVisible(false);
  };

  return (
    <>
      <TableRow
        css={{
          '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
          },
          height: theme.general.buttonHeight,
        }}
        data-testid="workspace-list-item"
      >
        <TableCell>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Link
              componentId="mlflow.home.workspaces.workspace_link"
              disableWorkspacePrefix
              to={`/?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(workspace.name)}`}
              onClick={(e) => {
                e.preventDefault();
                handleNameClick();
              }}
              css={{
                color: theme.colors.actionLinkDefault,
                textDecoration: 'none',
                fontWeight: theme.typography.typographyBoldFontWeight,
                '&:hover': {
                  color: theme.colors.actionLinkHover,
                  textDecoration: 'underline',
                },
              }}
            >
              {workspace.name}
            </Link>
            {isLastUsed && (
              <Tag componentId="mlflow.home.workspaces.last_used_badge" color="lemon">
                <FormattedMessage defaultMessage="Last used" description="Badge for last used workspace" />
              </Tag>
            )}
          </div>
        </TableCell>
        <TableCell>
          <div
            css={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              color: theme.colors.textSecondary,
            }}
          >
            {workspace.description}
          </div>
        </TableCell>
        {showEditColumn && (
          <TableCell css={{ width: 56, flex: '0 0 56px' }}>
            <div css={{ display: 'flex', justifyContent: 'center' }}>
              {canEdit && (
                <Button
                  componentId="mlflow.home.workspaces.edit_workspace"
                  size="small"
                  type="tertiary"
                  icon={<PencilIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Edit workspace',
                    description: 'Aria label for edit workspace button in workspaces table',
                  })}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleEditClick();
                  }}
                  css={{
                    opacity: 0,
                    '[role=row]:hover &': {
                      opacity: 1,
                    },
                    '[role=row]:focus-within &': {
                      opacity: 1,
                    },
                  }}
                />
              )}
            </div>
          </TableCell>
        )}
        {showManageColumn && (
          <TableCell
            css={{
              flex: 0,
              minWidth: 100,
              maxWidth: 100,
              justifyContent: 'center',
              paddingRight: theme.spacing.md,
            }}
          >
            {canManage ? (
              <Tooltip
                componentId="mlflow.home.workspaces.manage_tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Manage workspace',
                  description: 'Tooltip on the gear icon that links to /admin/ws scoped to this workspace',
                })}
              >
                <Link
                  componentId="mlflow.home.workspaces.manage_link"
                  disableWorkspacePrefix
                  to={`/admin/ws?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(workspace.name)}`}
                  onClick={(e) => {
                    e.preventDefault();
                    handleManageClick();
                  }}
                  aria-label={intl.formatMessage(
                    {
                      defaultMessage: 'Manage workspace {name}',
                      description: 'Aria label on the gear icon for managing a workspace',
                    },
                    { name: workspace.name },
                  )}
                  css={{
                    color: theme.colors.textSecondary,
                    display: 'inline-flex',
                    '&:hover': { color: theme.colors.actionLinkHover },
                  }}
                >
                  <GearIcon />
                </Link>
              </Tooltip>
            ) : null}
          </TableCell>
        )}
      </TableRow>

      <FormProvider {...form}>
        <Modal
          componentId="mlflow.home.workspaces.edit_modal"
          visible={isEditModalVisible}
          onCancel={handleCancel}
          onOk={handleSave}
          size="wide"
          okButtonProps={{ loading: isPending }}
          okText={intl.formatMessage({
            defaultMessage: 'Save',
            description: 'Save button text for edit workspace modal',
          })}
          cancelText={intl.formatMessage({
            defaultMessage: 'Cancel',
            description: 'Cancel button text for edit workspace modal',
          })}
          title={intl.formatMessage({
            defaultMessage: 'Edit Workspace',
            description: 'Title for edit workspace modal',
          })}
        >
          <div
            css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSave();
              }
            }}
          >
            {editError && (
              <Alert
                componentId="mlflow.home.workspaces.edit_modal.error"
                closable={false}
                message={editError}
                type="error"
              />
            )}
            <WorkspaceSettingsFields<EditWorkspaceFormData>
              idPrefix={`mlflow.home.workspaces.edit.${workspace.name}`}
              componentId="mlflow.home.workspaces.edit"
              fieldNames={{
                description: 'description',
                artifactRoot: 'defaultArtifactRoot',
                traceArchivalLocation: 'traceArchivalLocation',
              }}
              traceArchivalRetention={{
                amount: traceArchivalRetentionAmount,
                error: traceArchivalRetentionError,
                onAmountChange: (amount) => {
                  setDidEditTraceArchivalRetention(true);
                  updateTraceArchivalRetention({ amount });
                },
                onUnitChange: (unit) => {
                  setDidEditTraceArchivalRetention(true);
                  updateTraceArchivalRetention({ unit });
                },
                unit: traceArchivalRetentionUnit,
              }}
              descriptionAutoFocus
              showClearHint
              showTraceArchivalSettings={traceArchivalEnabled}
            />
          </div>
        </Modal>
      </FormProvider>
    </>
  );
};

const WORKSPACES_PER_PAGE = 10;

export const WorkspacesHomeView = ({ onCreateWorkspace }: WorkspacesHomeViewProps) => {
  const { theme } = useDesignSystemTheme();
  const { workspaces, isLoading, isError, refetch } = useWorkspaces(true);
  const traceArchivalEnabled = useTraceArchivalEnabled();
  const lastUsedWorkspace = getLastUsedWorkspace();
  const [currentPage, setCurrentPage] = useState(1);
  // Manage column: shown only to workspace managers — gear icon links to
  // ``/admin/ws?workspace=<name>`` (the per-workspace management view).
  // Platform admins navigate via the sidebar ``Manage`` entry instead,
  // which lands on ``/admin`` (the cross-workspace platform-admin view).
  //
  // ``useCurrentUserAdminWorkspaces`` already short-circuits to an empty set
  // for admins (it skips the role fetch when ``is_admin`` is true), so the
  // ``adminWorkspaces.size`` check alone would suffice — but explicitly
  // gating on ``!isAdmin`` is defense-in-depth so the gear stays hidden for
  // admins even if that short-circuit ever changes.
  const isAdmin = useCurrentUserIsAdmin();
  const isAuthAvailable = useIsAuthAvailable();
  const adminWorkspaces = useCurrentUserAdminWorkspaces();
  const canEditWorkspaces = !isAuthAvailable || isAdmin;
  const showEditColumn = canEditWorkspaces;
  const showManageColumn = !isAdmin && adminWorkspaces.size > 0;

  const shouldShowEmptyState = !isLoading && !isError && workspaces.length === 0;

  const paginatedWorkspaces = useMemo(() => {
    const startIndex = (currentPage - 1) * WORKSPACES_PER_PAGE;
    const endIndex = startIndex + WORKSPACES_PER_PAGE;
    return workspaces.slice(startIndex, endIndex);
  }, [workspaces, currentPage]);

  useEffect(() => {
    if (currentPage > 1 && paginatedWorkspaces.length === 0 && workspaces.length > 0) {
      setCurrentPage(1);
    }
  }, [workspaces.length, currentPage, paginatedWorkspaces.length]);

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.md,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Workspaces" description="Home page workspaces section title" />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Select a workspace to start experiments"
              description="Home page workspaces section subtitle"
            />
          </Typography.Text>
        </div>
        {!shouldShowEmptyState && (
          <Button componentId="mlflow.home.workspaces.create_button" onClick={onCreateWorkspace}>
            <FormattedMessage defaultMessage="Create new workspace" description="Create workspace button" />
          </Button>
        )}
      </div>
      <div
        css={{
          border: `1px solid ${theme.colors.border}`,
          overflow: 'hidden',
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        {isError ? (
          <div css={{ padding: theme.spacing.lg, display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Alert
              type="error"
              closable={false}
              componentId="mlflow.home.workspaces.error"
              message={
                <FormattedMessage
                  defaultMessage="We couldn't load your workspaces."
                  description="Home page workspaces error message"
                />
              }
            />
            <div>
              <Button componentId="mlflow.home.workspaces.retry" onClick={() => refetch()}>
                <FormattedMessage defaultMessage="Retry" description="Home page workspaces retry CTA" />
              </Button>
            </div>
          </div>
        ) : shouldShowEmptyState ? (
          <WorkspacesEmptyState onCreateWorkspace={onCreateWorkspace} />
        ) : (
          <Table>
            <TableRow isHeader>
              <TableHeader componentId="mlflow.home.workspaces_table.header.name">
                <FormattedMessage defaultMessage="Name" description="Workspaces table name column header" />
              </TableHeader>
              <TableHeader componentId="mlflow.home.workspaces_table.header.description">
                <FormattedMessage
                  defaultMessage="Description"
                  description="Workspaces table description column header"
                />
              </TableHeader>
              {showEditColumn && (
                <TableHeader
                  componentId="mlflow.home.workspaces_table.header.actions"
                  css={{ width: 56, flex: '0 0 56px' }}
                />
              )}
              {showManageColumn && (
                <TableHeader
                  componentId="mlflow.home.workspaces_table.header.manage"
                  css={{
                    flex: 0,
                    minWidth: 100,
                    maxWidth: 100,
                    justifyContent: 'center',
                    paddingRight: theme.spacing.md,
                  }}
                >
                  <FormattedMessage
                    defaultMessage="Manage"
                    description="Workspaces table column header for the per-row admin entry point"
                  />
                </TableHeader>
              )}
            </TableRow>
            {isLoading ? (
              <TableRow>
                <TableCell />
                <TableCell css={{ padding: theme.spacing.lg, textAlign: 'center' }}>
                  <FormattedMessage defaultMessage="Loading workspaces..." description="Loading workspaces message" />
                </TableCell>
                {showEditColumn && <TableCell css={{ width: 56, flex: '0 0 56px' }} />}
                {showManageColumn && <TableCell />}
              </TableRow>
            ) : (
              paginatedWorkspaces.map((workspace) => (
                <WorkspaceRow
                  key={workspace.name}
                  workspace={workspace}
                  isLastUsed={workspace.name === lastUsedWorkspace}
                  canEdit={canEditWorkspaces}
                  canManage={!isAdmin && adminWorkspaces.has(workspace.name)}
                  showEditColumn={showEditColumn}
                  showManageColumn={showManageColumn}
                  traceArchivalEnabled={traceArchivalEnabled}
                />
              ))
            )}
          </Table>
        )}
        {!shouldShowEmptyState && !isLoading && !isError && workspaces.length > WORKSPACES_PER_PAGE && (
          <div
            css={{
              padding: theme.spacing.md,
              display: 'flex',
              justifyContent: 'center',
              borderTop: `1px solid ${theme.colors.border}`,
            }}
          >
            <Pagination
              componentId="mlflow.home.workspaces.pagination"
              currentPageIndex={currentPage}
              numTotal={workspaces.length}
              pageSize={WORKSPACES_PER_PAGE}
              onChange={(newPageNumber) => setCurrentPage(newPageNumber)}
            />
          </div>
        )}
      </div>
      <Spacer shrinks={false} />
    </section>
  );
};

export default WorkspacesHomeView;
