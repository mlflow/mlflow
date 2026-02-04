import { useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Input,
  Modal,
  Pagination,
  PencilIcon,
  Spacer,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
  Tag,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link, useNavigate } from '../../common/utils/RoutingUtils';
import { useWorkspaces, type Workspace } from '../../workspaces/hooks/useWorkspaces';
import { getLastUsedWorkspace, setActiveWorkspace, WORKSPACE_QUERY_PARAM } from '../../workspaces/utils/WorkspaceUtils';
import { useUpdateWorkspace } from '../../workspaces/hooks/useUpdateWorkspace';
import Utils from '../../common/utils/Utils';

type WorkspacesHomeViewProps = {
  onCreateWorkspace: () => void;
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

const WorkspaceRow = ({ workspace, isLastUsed }: { workspace: Workspace; isLastUsed: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const intl = useIntl();
  const { mutate: updateWorkspace, isLoading: isPending } = useUpdateWorkspace();

  const [editingField, setEditingField] = useState<'description' | 'artifact_root' | null>(null);
  const [editValue, setEditValue] = useState('');

  const handleNameClick = () => {
    setActiveWorkspace(workspace.name);
    // Navigate to the home page with workspace query param
    navigate(`/?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(workspace.name)}`);
  };

  const handleEditClick = (field: 'description' | 'artifact_root', currentValue: string | null | undefined) => {
    setEditingField(field);
    setEditValue(currentValue || '');
  };

  const handleSave = () => {
    if (editingField === null) return;

    const updateData: { name: string; description?: string; default_artifact_root?: string } = {
      name: workspace.name,
    };

    if (editingField === 'description') {
      updateData.description = editValue;
    } else if (editingField === 'artifact_root') {
      updateData.default_artifact_root = editValue;
    }

    updateWorkspace(updateData, {
      onSuccess: () => {
        setEditingField(null);
        setEditValue('');
      },
      onError: (error: any) => {
        // Display error notification to user
        Utils.logErrorAndNotifyUser(error);
      },
    });
  };

  const handleCancel = () => {
    setEditingField(null);
    setEditValue('');
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
          <div css={{ display: 'flex' }}>
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
            <Button
              componentId={`mlflow.home.workspaces.edit_description.${workspace.name}`}
              size="small"
              type="tertiary"
              icon={workspace.description ? <PencilIcon /> : undefined}
              onClick={(e) => {
                e.stopPropagation();
                handleEditClick('description', workspace.description);
              }}
              aria-label={intl.formatMessage({
                defaultMessage: 'Edit description',
                description: 'Label for edit description button in workspaces table',
              })}
              css={{
                flexShrink: 0,
                opacity: 0,
                '[role=row]:hover &': {
                  opacity: 1,
                },
                '[role=row]:focus-within &': {
                  opacity: 1,
                },
              }}
            >
              {!workspace.description ? (
                <FormattedMessage
                  defaultMessage="Set description"
                  description="Label for set description button in workspaces table"
                />
              ) : undefined}
            </Button>
          </div>
        </TableCell>
        <TableCell>
          <div css={{ display: 'flex' }}>
            <div
              css={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                color: theme.colors.textSecondary,
              }}
            >
              {workspace.default_artifact_root}
            </div>
            <Button
              componentId={`mlflow.home.workspaces.edit_artifact_root.${workspace.name}`}
              size="small"
              type="tertiary"
              icon={workspace.default_artifact_root ? <PencilIcon /> : undefined}
              onClick={(e) => {
                e.stopPropagation();
                handleEditClick('artifact_root', workspace.default_artifact_root);
              }}
              aria-label={intl.formatMessage({
                defaultMessage: 'Edit artifact root',
                description: 'Label for edit artifact root button in workspaces table',
              })}
              css={{
                flexShrink: 0,
                opacity: 0,
                '[role=row]:hover &': {
                  opacity: 1,
                },
                '[role=row]:focus-within &': {
                  opacity: 1,
                },
              }}
            >
              {!workspace.default_artifact_root ? (
                <FormattedMessage
                  defaultMessage="Set artifact root"
                  description="Label for set artifact root button in workspaces table"
                />
              ) : undefined}
            </Button>
          </div>
        </TableCell>
      </TableRow>

      <Modal
        componentId={`mlflow.home.workspaces.edit_modal.${workspace.name}`}
        visible={editingField !== null}
        onCancel={handleCancel}
        onOk={handleSave}
        okButtonProps={{ loading: isPending }}
        okText={intl.formatMessage({
          defaultMessage: 'Save',
          description: 'Save button text for edit workspace modal',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel button text for edit workspace modal',
        })}
        title={
          editingField === 'description'
            ? intl.formatMessage({
                defaultMessage: 'Edit Description',
                description: 'Title for edit workspace description modal',
              })
            : intl.formatMessage({
                defaultMessage: 'Edit Artifact Root',
                description: 'Title for edit workspace artifact root modal',
              })
        }
      >
        <div
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSave();
            }
          }}
        >
          <Input
            componentId={`mlflow.home.workspaces.edit_${editingField}_input`}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            placeholder={
              editingField === 'description'
                ? intl.formatMessage({
                    defaultMessage: 'Enter description',
                    description: 'Placeholder for description input in edit modal',
                  })
                : intl.formatMessage({
                    defaultMessage: 'Enter artifact root URI',
                    description: 'Placeholder for artifact root input in edit modal',
                  })
            }
            autoFocus
          />
        </div>
      </Modal>
    </>
  );
};

const WORKSPACES_PER_PAGE = 10;

export const WorkspacesHomeView = ({ onCreateWorkspace }: WorkspacesHomeViewProps) => {
  const { theme } = useDesignSystemTheme();
  const { workspaces, isLoading, isError, refetch } = useWorkspaces(true);
  // Get last used workspace from localStorage for the "Last used" badge
  const lastUsedWorkspace = getLastUsedWorkspace();
  const [currentPage, setCurrentPage] = useState(1);

  const shouldShowEmptyState = !isLoading && !isError && workspaces.length === 0;

  // Calculate paginated workspaces
  const paginatedWorkspaces = useMemo(() => {
    const startIndex = (currentPage - 1) * WORKSPACES_PER_PAGE;
    const endIndex = startIndex + WORKSPACES_PER_PAGE;
    return workspaces.slice(startIndex, endIndex);
  }, [workspaces, currentPage]);

  // Reset to page 1 when workspaces change
  useMemo(() => {
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
              <TableHeader componentId="mlflow.home.workspaces_table.header.artifact_root">
                <FormattedMessage
                  defaultMessage="Artifact Root"
                  description="Workspaces table artifact root column header"
                />
              </TableHeader>
            </TableRow>
            {isLoading ? (
              <TableRow>
                <TableCell />
                <TableCell css={{ padding: theme.spacing.lg, textAlign: 'center' }}>
                  <FormattedMessage defaultMessage="Loading workspaces..." description="Loading workspaces message" />
                </TableCell>
                <TableCell />
              </TableRow>
            ) : (
              paginatedWorkspaces.map((workspace) => (
                <WorkspaceRow
                  key={workspace.name}
                  workspace={workspace}
                  isLastUsed={workspace.name === lastUsedWorkspace}
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
