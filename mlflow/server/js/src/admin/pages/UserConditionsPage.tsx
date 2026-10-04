import { useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Link, useParams } from '../../common/utils/RoutingUtils';
import AdminRoutes from '../routes';
import { useUserMutationConditionsQuery, useWithSettingsReturnTo } from '../hooks';
import { isSyntheticUserRole } from '../../account/types';
import { formatConditionScope, getResourceTypeLabel } from '../types';
import { EditAccessModal } from '../components/EditAccessModal';

/**
 * Every condition that currently restricts one user, grouped by the role carrying it.
 *
 * There is no user-level condition store, and the grouping is not just a consequence of
 * that -- it is the answer to the question an admin brings to this page. A user whose
 * write was refused needs to know which role is responsible, because that is the only
 * place the restriction can be changed.
 *
 * Read-only by design: a condition belongs to a role, so editing one here would edit it
 * for every user holding that role. The per-role link is the edit path. What this page
 * does offer is role assignment, since that is the one thing about a user's conditions
 * that is genuinely a property of the user.
 */
const UserConditionsPage = () => {
  const { theme } = useDesignSystemTheme();
  const { username: usernameParam = '' } = useParams<{ username: string }>();
  // Routes encode the username because it may contain ``/`` or ``%``.
  const username = decodeURIComponent(usernameParam);
  const withReturnTo = useWithSettingsReturnTo();
  const [editAccessOpen, setEditAccessOpen] = useState(false);

  const { groups, isLoading, error, totalConditions } = useUserMutationConditionsQuery(username);

  if (!username) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="admin.user_conditions.invalid_username"
            type="error"
            message="Invalid username"
            description="The requested user could not be loaded because the URL contains no username."
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  // Groups with no conditions are dropped: a role that restricts nothing is not part of
  // the answer to "what is restricting this user?", and listing every role would bury
  // the few that matter.
  const restrictingGroups = groups.filter((g) => g.conditions.length > 0);

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link componentId="admin.user_conditions.breadcrumb_admin" to={withReturnTo(AdminRoutes.adminPageRoute)}>
                Platform Admin
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link
                componentId="admin.user_conditions.breadcrumb_user"
                to={withReturnTo(AdminRoutes.getUserDetailRoute(username))}
              >
                {username}
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>Conditions</Breadcrumb.Item>
          </Breadcrumb>
          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Title withoutMargins level={2}>
                Conditions for {username}
              </Typography.Title>
              <Typography.Text color="secondary">
                Conditions come from this user's roles. They narrow what the role's grants allow and never grant
                anything on their own.
              </Typography.Text>
            </div>
            <Button
              componentId="admin.user_conditions.manage_roles_button"
              type="primary"
              onClick={() => setEditAccessOpen(true)}
            >
              Manage roles
            </Button>
          </div>
        </div>

        <Alert
          componentId="admin.user_conditions.semantics_notice"
          type="info"
          closable={false}
          message="Adding a role cannot lift a condition"
          description={
            <span>
              Every condition below must pass for a write to be allowed, so assigning another role adds its grants but
              also adds its conditions. To remove a restriction, edit it on the role that carries it. Admins and
              workspace managers bypass conditions entirely.
            </span>
          }
        />

        {error && (
          <Alert
            componentId="admin.user_conditions.fetch_error"
            type="error"
            message="Failed to load conditions"
            description={(error as Error)?.message || 'An error occurred while fetching conditions.'}
          />
        )}

        {isLoading ? (
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg, minHeight: 200 }}>
            <Spinner size="small" />
          </div>
        ) : restrictingGroups.length === 0 ? (
          <Empty
            title="No conditions"
            description={
              totalConditions === 0 && groups.length === 0
                ? 'This user holds no roles, so nothing restricts them beyond their grants.'
                : "None of this user's roles carry a condition, so their grants apply in full."
            }
          />
        ) : (
          restrictingGroups.map((group) => (
            <div key={group.role.id} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <Typography.Title withoutMargins level={4}>
                  {/* A direct grant is backed by a synthetic role. Name it for what it is
                      rather than leaking ``__user_<id>__``, but do not hide it: a
                      condition sitting on it is a live restriction. */}
                  {isSyntheticUserRole(group.role.name) ? 'Direct grants' : group.role.name}
                </Typography.Title>
                <Tag componentId="admin.user_conditions.count_tag" color="indigo">
                  {group.conditions.length}
                </Tag>
                {!isSyntheticUserRole(group.role.name) && (
                  <Link
                    componentId="admin.user_conditions.edit_on_role"
                    to={withReturnTo(AdminRoutes.getRoleConditionsRoute(group.role.id))}
                  >
                    Edit on role
                  </Link>
                )}
              </div>
              <Table
                scrollable
                noMinHeight
                css={{
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.general.borderRadiusBase,
                  overflow: 'hidden',
                }}
              >
                <TableRow isHeader>
                  <TableHeader componentId="admin.user_conditions.type_header" css={{ flex: 1 }}>
                    Resource type
                  </TableHeader>
                  <TableHeader componentId="admin.user_conditions.scope_header" css={{ flex: 1 }}>
                    Scope
                  </TableHeader>
                  <TableHeader componentId="admin.user_conditions.request_header" css={{ flex: 2 }}>
                    Request filter
                  </TableHeader>
                  <TableHeader componentId="admin.user_conditions.resource_header" css={{ flex: 2 }}>
                    Resource filter
                  </TableHeader>
                </TableRow>
                {group.conditions.map((condition) => (
                  <TableRow key={condition.id}>
                    <TableCell css={{ flex: 1 }}>
                      <Tag componentId="admin.user_conditions.type_tag">
                        {getResourceTypeLabel(condition.resource_type)}
                      </Tag>
                    </TableCell>
                    <TableCell css={{ flex: 1 }}>
                      <Typography.Text size="sm">{formatConditionScope(condition)}</Typography.Text>
                    </TableCell>
                    <TableCell css={{ flex: 2 }}>
                      {condition.value_condition ? (
                        <code>{condition.value_condition}</code>
                      ) : (
                        <Typography.Text color="secondary" size="sm">
                          —
                        </Typography.Text>
                      )}
                    </TableCell>
                    <TableCell css={{ flex: 2 }}>
                      {condition.target_condition ? (
                        <code>{condition.target_condition}</code>
                      ) : (
                        <Typography.Text color="secondary" size="sm">
                          —
                        </Typography.Text>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </Table>
            </div>
          ))
        )}

        <EditAccessModal open={editAccessOpen} onClose={() => setEditAccessOpen(false)} username={username} />
      </div>
    </ScrollablePageWrapper>
  );
};

export default UserConditionsPage;
