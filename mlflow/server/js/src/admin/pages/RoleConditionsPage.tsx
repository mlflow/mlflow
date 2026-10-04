import { useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  CloseIcon,
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
import {
  useAddMutationCondition,
  useRemoveMutationCondition,
  useRoleDetailQuery,
  useRoleMutationConditionsQuery,
  useWithSettingsReturnTo,
} from '../hooks';
import { formatConditionScope, getResourceTypeLabel, MAX_CONDITIONS_PER_ROLE_TYPE } from '../types';
import { FieldLabel } from '../components/FieldLabel';
import {
  draftToAddRequest,
  isMutationConditionDraftDirty,
  isMutationConditionDraftFillable,
  MUTATION_CONDITION_DRAFT_DEFAULT,
  MutationConditionForm,
  type MutationConditionDraft,
} from '../components/MutationConditionForm';

/**
 * A role's mutation conditions.
 *
 * Its own page rather than a third tab on the role detail, for the reason the page body
 * states: a condition is not another kind of grant. A grant on the Permissions tab
 * widens what the role can do; a condition narrows it, and every condition the user
 * holds across every role has to pass. Putting them side by side as peer tabs would
 * suggest they combine, and the direction of a condition is the one thing an admin must
 * not get backwards.
 *
 * Unlike the grants flow -- which stages a list and submits it with the role -- each
 * condition is added and removed on its own, because the server addresses conditions by
 * their own id and allocates a slot per add. Staging would buy nothing and would have to
 * invent client-side ordering the server owns.
 */
const RoleConditionsPage = () => {
  const { theme } = useDesignSystemTheme();
  const { roleId: roleIdParam } = useParams<{ roleId: string }>();
  const roleId = Number(roleIdParam);
  const isValidRoleId = Number.isFinite(roleId);
  const withReturnTo = useWithSettingsReturnTo();

  const { data: roleData, isLoading: roleLoading, error: roleError } = useRoleDetailQuery(roleId);
  const {
    data: conditionsData,
    isLoading: conditionsLoading,
    error: conditionsError,
  } = useRoleMutationConditionsQuery(roleId, { enabled: isValidRoleId });

  const addCondition = useAddMutationCondition(roleId);
  const removeCondition = useRemoveMutationCondition(roleId);

  const [draft, setDraft] = useState<MutationConditionDraft>(MUTATION_CONDITION_DRAFT_DEFAULT);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const role = roleData?.role;
  const conditions = conditionsData?.mutation_conditions ?? [];

  const dirty = isMutationConditionDraftDirty(draft);
  const canAdd = isMutationConditionDraftFillable(draft);
  // Narrow the two inline reminders to the specific thing that is missing, so a draft
  // that cannot be submitted says which field is at fault rather than just refusing.
  const showFilterRequired = dirty && !draft.valueCondition.trim() && !draft.targetCondition.trim();
  const showParentRequired = dirty && draft.scope === 'parent' && !draft.parentResourceId.trim();

  // The cap is per (role, resource type), not per role, so count only the type in play.
  const sameTypeCount = conditions.filter((c) => c.resource_type === draft.resourceType).length;
  const atCap = sameTypeCount >= MAX_CONDITIONS_PER_ROLE_TYPE;

  const handleAdd = async () => {
    if (!canAdd || atCap) return;
    setSubmitError(null);
    try {
      await addCondition.mutateAsync(draftToAddRequest(draft, roleId));
      setDraft(MUTATION_CONDITION_DRAFT_DEFAULT);
    } catch (e) {
      // Surface the server's message verbatim: a rejected filter string is reported with
      // the parse error, which is the only useful guidance for fixing it.
      setSubmitError((e as Error)?.message || 'Failed to add the condition.');
    }
  };

  const handleRemove = async (conditionId: number) => {
    setSubmitError(null);
    try {
      await removeCondition.mutateAsync(conditionId);
    } catch (e) {
      setSubmitError((e as Error)?.message || 'Failed to remove the condition.');
    }
  };

  if (!isValidRoleId) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="admin.role_conditions.invalid_id"
            type="error"
            message="Invalid role ID"
            description="The requested role could not be loaded because the URL contains an invalid role ID."
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (roleLoading) {
    return (
      <ScrollablePageWrapper>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: theme.spacing.lg,
            minHeight: 200,
          }}
        >
          <Spinner size="small" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (roleError || !role) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="admin.role_conditions.load_error"
            type="error"
            message={(roleError as Error)?.message || 'Role not found'}
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  const emptyState =
    conditions.length === 0 ? (
      <Empty
        title="No conditions"
        description="This role's grants apply in full. Add a condition below to narrow them."
      />
    ) : null;

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link
                componentId="admin.role_conditions.breadcrumb_admin"
                to={withReturnTo(`${AdminRoutes.adminPageRoute}?tab=roles`)}
              >
                Platform Admin
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link
                componentId="admin.role_conditions.breadcrumb_role"
                to={withReturnTo(AdminRoutes.getRoleDetailRoute(roleId))}
              >
                {role.name}
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>Conditions</Breadcrumb.Item>
          </Breadcrumb>
          <Typography.Title withoutMargins level={2}>
            Conditions for {role.name}
          </Typography.Title>
          <Typography.Text color="secondary">
            A condition narrows what this role's grants allow. It never grants anything on its own, and it only applies
            to writes — reads are never affected.
          </Typography.Text>
        </div>

        <Alert
          componentId="admin.role_conditions.semantics_notice"
          type="info"
          closable={false}
          message="Conditions subtract, and they stack"
          description={
            <span>
              Every condition that applies must pass, including those carried by the user's other roles. Adding a second
              role cannot lift a restriction set by the first. Admins and workspace managers bypass conditions entirely.
            </span>
          }
        />

        {submitError && (
          <Alert
            componentId="admin.role_conditions.submit_error"
            type="error"
            message={submitError}
            onClose={() => setSubmitError(null)}
          />
        )}

        {conditionsError && (
          <Alert
            componentId="admin.role_conditions.fetch_error"
            type="error"
            message="Failed to load conditions"
            description={(conditionsError as Error)?.message || 'An error occurred while fetching conditions.'}
          />
        )}

        {conditionsLoading ? (
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
            <Spinner size="small" />
          </div>
        ) : (
          <Table
            scrollable
            noMinHeight
            empty={emptyState}
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.general.borderRadiusBase,
              overflow: 'hidden',
            }}
          >
            <TableRow isHeader>
              <TableHeader componentId="admin.role_conditions.type_header" css={{ flex: 1 }}>
                Resource type
              </TableHeader>
              <TableHeader componentId="admin.role_conditions.scope_header" css={{ flex: 1 }}>
                Scope
              </TableHeader>
              <TableHeader componentId="admin.role_conditions.request_header" css={{ flex: 2 }}>
                Request filter
              </TableHeader>
              <TableHeader componentId="admin.role_conditions.resource_header" css={{ flex: 2 }}>
                Resource filter
              </TableHeader>
              <TableHeader
                componentId="admin.role_conditions.actions_header"
                css={{ flex: 0, minWidth: 60, maxWidth: 60 }}
              >
                {' '}
              </TableHeader>
            </TableRow>
            {conditions.map((condition) => (
              <TableRow key={condition.id}>
                <TableCell css={{ flex: 1 }}>
                  <Tag componentId="admin.role_conditions.type_tag">
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
                <TableCell css={{ flex: 0, minWidth: 60, maxWidth: 60 }}>
                  <Button
                    componentId="admin.role_conditions.remove"
                    type="tertiary"
                    size="small"
                    icon={<CloseIcon />}
                    aria-label={`Remove ${getResourceTypeLabel(condition.resource_type)} condition`}
                    onClick={() => handleRemove(condition.id)}
                    loading={removeCondition.isLoading}
                  />
                </TableCell>
              </TableRow>
            ))}
          </Table>
        )}

        <div
          css={{
            border: `1px dashed ${theme.colors.border}`,
            borderRadius: theme.general.borderRadiusBase,
            padding: theme.spacing.md,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <FieldLabel>Add a condition</FieldLabel>
          <MutationConditionForm
            value={draft}
            onChange={setDraft}
            workspace={role.workspace}
            disabled={addCondition.isLoading}
            showFilterRequiredError={showFilterRequired}
            showParentRequiredError={showParentRequired}
          />
          {atCap && (
            <Typography.Text color="error" size="sm" data-testid="admin.role_conditions.at_cap">
              This role already has the maximum of {MAX_CONDITIONS_PER_ROLE_TYPE} conditions for{' '}
              {getResourceTypeLabel(draft.resourceType).toLowerCase()}. Remove one before adding another.
            </Typography.Text>
          )}
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
            {dirty && (
              <Button
                componentId="admin.role_conditions.clear"
                type="tertiary"
                onClick={() => {
                  setDraft(MUTATION_CONDITION_DRAFT_DEFAULT);
                  setSubmitError(null);
                }}
                disabled={addCondition.isLoading}
              >
                Clear
              </Button>
            )}
            <Button
              componentId="admin.role_conditions.add"
              type="primary"
              onClick={handleAdd}
              disabled={!canAdd || atCap || addCondition.isLoading}
              loading={addCondition.isLoading}
            >
              Add condition
            </Button>
          </div>
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default RoleConditionsPage;
