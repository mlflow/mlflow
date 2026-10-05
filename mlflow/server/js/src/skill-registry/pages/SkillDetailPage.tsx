import { useMemo, useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  DropdownMenu,
  GenericSkeleton,
  Header,
  LockIcon,
  OverflowIcon,
  Spacer,
  TableSkeleton,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { Link, useNavigate, useParams } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import SkillRegistryRoutes from '../routes';
import {
  formatSkillOrganization,
  getSkillPermissions,
  isNotFoundError,
  isPermissionDeniedError,
  isSkillDimmed,
  parseSkillRouteParams,
} from '../utils';
import { headerIconStyles } from '../styles';
import { useSkillQuery } from '../hooks/useSkillQuery';
import { useSkillVersionSelection } from '../hooks/useSkillVersionSelection';
import { SkillIcon } from '../components/SkillIcon';
import { SkillTags } from '../components/SkillTags';
import { SkillExpandableDescription } from '../components/SkillExpandableDescription';
import { SkillPencilButton } from '../components/SkillPencilButton';
import { SkillVersionList } from '../components/SkillVersionList';
import { SkillVersionDetail } from '../components/SkillVersionDetail';
import { RegisterSkillModal } from '../components/RegisterSkillModal';
import { useEditSkillModal } from '../hooks/useEditSkillModal';
import { useDeleteSkillModal } from '../hooks/useDeleteSkillModal';
import { useDeleteSkillVersionModal } from '../hooks/useDeleteSkillVersionModal';
import { useSkillMetadataEditors } from '../hooks/useSkillMetadataEditors';
import { SkillRegistryEmptyState } from '../components/SkillRegistryEmptyState';
import type { Skill } from '../types';

const breadcrumbs = (
  <Breadcrumb>
    <Breadcrumb.Item>
      <Link componentId="mlflow.skill_registry.detail.breadcrumb_back" to={SkillRegistryRoutes.skillRegistryPageRoute}>
        <FormattedMessage defaultMessage="Skills" description="Breadcrumb link back to the Skill Registry catalog" />
      </Link>
    </Breadcrumb.Item>
  </Breadcrumb>
);

const SkillDetailHeader = ({
  skill,
  canUpdate,
  onCreateVersion,
  onEdit,
  onEditTags,
  onDelete,
}: {
  skill: Skill;
  canUpdate: boolean;
  onCreateVersion: () => void;
  onEdit: () => void;
  onEditTags: () => void;
  onDelete?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isDimmed = isSkillDimmed(skill);
  const organizationLabel = formatSkillOrganization(skill.organization);
  const hasTags = Object.keys(skill.tags || {}).length > 0;

  return (
    <>
      <Header
        breadcrumbs={breadcrumbs}
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span css={headerIconStyles(theme)}>
              <SkillIcon icons={skill.icons} name={skill.name} />
            </span>
            {skill.name}
            {isDimmed && (
              <Tooltip
                componentId="mlflow.skill_registry.detail.unavailable_tooltip"
                content={
                  <FormattedMessage
                    defaultMessage="Set the skill status to active to make it available"
                    description="Tooltip for unavailable label on Skill detail page"
                  />
                }
              >
                <span css={{ cursor: 'default' }}>
                  <Tag componentId="mlflow.skill_registry.detail.unavailable_tag" color="coral">
                    <FormattedMessage
                      defaultMessage="Unavailable"
                      description="Label for a Skill whose derived parent status is not active"
                    />
                  </Tag>
                </span>
              </Tooltip>
            )}
          </span>
        }
        buttons={
          <>
            {(canUpdate || onDelete) && (
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <Button
                    componentId="mlflow.skill_registry.detail.actions"
                    icon={<OverflowIcon />}
                    aria-label={intl.formatMessage({
                      defaultMessage: 'More actions',
                      description: 'Aria label for skill detail actions menu',
                    })}
                  />
                </DropdownMenu.Trigger>
                <DropdownMenu.Content>
                  {canUpdate && (
                    <DropdownMenu.Item componentId="mlflow.skill_registry.detail.actions.edit" onClick={onEdit}>
                      <FormattedMessage
                        defaultMessage="Edit"
                        description="Skill detail action that edits description and icons"
                      />
                    </DropdownMenu.Item>
                  )}
                  {onDelete && (
                    <DropdownMenu.Item componentId="mlflow.skill_registry.detail.actions.delete" onClick={onDelete}>
                      <FormattedMessage
                        defaultMessage="Delete"
                        description="Skill detail action that deletes the skill"
                      />
                    </DropdownMenu.Item>
                  )}
                </DropdownMenu.Content>
              </DropdownMenu.Root>
            )}
            {canUpdate && (
              <Button componentId="mlflow.skill_registry.create_version" type="primary" onClick={onCreateVersion}>
                <FormattedMessage
                  defaultMessage="Create skill version"
                  description="Button that adds an external-source version to this skill"
                />
              </Button>
            )}
          </>
        }
      />
      {organizationLabel && (
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
          {organizationLabel}
        </Typography.Text>
      )}
      {skill.description && <SkillExpandableDescription key={skill.description} text={skill.description} />}
      {(hasTags || canUpdate) && (
        <div
          css={{
            marginTop: theme.spacing.xs,
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: theme.spacing.xs,
          }}
        >
          {hasTags && <SkillTags tags={skill.tags} wrap />}
          {canUpdate && (
            <SkillPencilButton
              componentId="mlflow.skill_registry.detail.tags.edit"
              label={intl.formatMessage({
                defaultMessage: 'Edit tags',
                description: 'Aria label for editing skill parent tags',
              })}
              onClick={onEditTags}
            />
          )}
        </div>
      )}
    </>
  );
};

const SkillDetailPage = () => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const params = useParams<{ skillKey?: string; organization?: string; skillName?: string }>();
  const { name, organization } = parseSkillRouteParams(params);
  const [createVersionOpen, setCreateVersionOpen] = useState(false);
  const { data: skill, isLoading: skillLoading, error: skillError, refetch } = useSkillQuery(name, organization);
  const {
    versionsQuery: { data: versions, hasMoreVersions, isLoading: versionsLoading, error: versionsError },
    selectedVersion,
    setSelectedVersion,
    currentVersion,
    isLoading: isVersionDetailLoading,
    isMissing: selectedVersionMissing,
    error: selectedVersionError,
  } = useSkillVersionSelection(name, organization, skill);

  const permissions = getSkillPermissions(skill);
  // Kept stable so the shared alias editor's memoization survives unrelated page renders.
  const aliases = useMemo(
    () =>
      (skill?.aliases ?? [])
        .filter((alias) => alias.alias !== 'latest')
        .map((alias) => ({ alias: alias.alias, version: String(alias.version) })),
    [skill?.aliases],
  );
  const { EditSkillModal, openEditSkill } = useEditSkillModal({ name, organization });
  const { DeleteSkillModal, openDeleteSkill } = useDeleteSkillModal({
    name,
    organization,
    onDeleted: () => navigate(SkillRegistryRoutes.skillRegistryPageRoute),
  });
  const { DeleteSkillVersionModal, openDeleteSkillVersion } = useDeleteSkillVersionModal({
    name,
    organization,
    onDeleted: (version) => {
      const next = versions?.find((candidate) => candidate.version !== version);
      setSelectedVersion(next?.version);
    },
  });
  const governance = useSkillMetadataEditors({
    name,
    organization,
    aliases,
    canDelete: permissions.canDelete,
  });
  const { statusMutation } = governance;
  const editableVersion = permissions.canUpdate ? currentVersion : undefined;
  const deletableVersion = permissions.canDelete ? currentVersion : undefined;

  if (skillLoading) {
    return (
      <ScrollablePageWrapper>
        <Spacer shrinks={false} />
        <Header
          breadcrumbs={breadcrumbs}
          title={<GenericSkeleton css={{ height: theme.general.heightBase, width: 200 }} />}
        />
        <Spacer shrinks={false} />
        <div css={{ display: 'flex', gap: theme.spacing.lg }}>
          <div css={{ flex: '0 0 320px' }}>
            <TableSkeleton lines={6} />
          </div>
          <div css={{ flex: 1 }}>
            <TableSkeleton lines={4} />
          </div>
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (isPermissionDeniedError(skillError)) {
    return (
      <ScrollablePageWrapper>
        <Spacer shrinks={false} />
        <Header breadcrumbs={breadcrumbs} title="" />
        <SkillRegistryEmptyState
          image={<LockIcon />}
          title={
            <FormattedMessage
              defaultMessage="Permission denied"
              description="Title for Skill detail permission-denied state"
            />
          }
          description={
            skillError?.message || (
              <FormattedMessage
                defaultMessage="You do not have permission to view this skill."
                description="Description for Skill detail permission-denied state"
              />
            )
          }
        />
      </ScrollablePageWrapper>
    );
  }

  if (skillError || !skill) {
    const missing = isNotFoundError(skillError) || !skillError;
    return (
      <ScrollablePageWrapper>
        <Spacer shrinks={false} />
        <Header breadcrumbs={breadcrumbs} title="" />
        {missing ? (
          <SkillRegistryEmptyState
            title={
              <FormattedMessage defaultMessage="Skill not found" description="Title when a Skill cannot be loaded" />
            }
            description={
              skillError?.message || (
                <FormattedMessage
                  defaultMessage="This skill may have been deleted or the URL may be incorrect."
                  description="Description when a Skill cannot be loaded"
                />
              )
            }
          />
        ) : (
          <Alert
            componentId="mlflow.skill_registry.detail.error"
            type="error"
            message={
              <FormattedMessage defaultMessage="Failed to load skill" description="Skill detail page error title" />
            }
            description={skillError?.message}
            closable={false}
            actions={[
              {
                componentId: 'mlflow.skill_registry.detail.error.retry',
                children: (
                  <FormattedMessage defaultMessage="Retry" description="Retry button for Skill detail load error" />
                ),
                onClick: () => refetch(),
              },
            ]}
          />
        )}
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <SkillDetailHeader
        skill={skill}
        canUpdate={permissions.canUpdate}
        onCreateVersion={() => setCreateVersionOpen(true)}
        onEdit={() => openEditSkill(skill)}
        onEditTags={() => governance.showEditParentTags(skill)}
        onDelete={permissions.canDelete ? openDeleteSkill : undefined}
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div css={{ flex: '0 0 320px', display: 'flex', flexDirection: 'column' }}>
          {versionsError ? (
            <Alert
              componentId="mlflow.skill_registry.detail.versions_error"
              type="error"
              message={versionsError.message}
              closable={false}
            />
          ) : (
            <SkillVersionList
              versions={versions}
              selectedVersion={selectedVersion}
              onSelectVersion={setSelectedVersion}
              isLoading={versionsLoading}
              hasMoreVersions={hasMoreVersions}
            />
          )}
        </div>
        <div
          css={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            minWidth: 0,
            borderLeft: `1px solid ${theme.colors.border}`,
            overflow: 'hidden',
          }}
        >
          <SkillVersionDetail
            skill={skill}
            version={currentVersion}
            isLoading={isVersionDetailLoading}
            isMissing={selectedVersionMissing}
            error={selectedVersionError}
            onEditAliases={editableVersion && (() => governance.showEditAliasesModal(String(editableVersion.version)))}
            onEditMetadata={editableVersion && (() => governance.showEditVersionMetadata(editableVersion))}
            onDelete={deletableVersion && (() => openDeleteSkillVersion(deletableVersion.version))}
            statusUpdate={
              editableVersion && {
                onChange: (status) => statusMutation.mutate({ version: editableVersion.version, status }),
                isPending: statusMutation.isLoading,
                error: statusMutation.variables?.version === editableVersion.version ? statusMutation.error : undefined,
                onDismissError: statusMutation.reset,
              }
            }
          />
        </div>
      </div>
      <RegisterSkillModal
        visible={createVersionOpen}
        skill={{ name: skill.name, organization: skill.organization }}
        sourceVersion={currentVersion}
        onClose={() => setCreateVersionOpen(false)}
        onRegistered={(version) => setSelectedVersion(version.version)}
      />
      {EditSkillModal}
      {DeleteSkillModal}
      {DeleteSkillVersionModal}
      {governance.EditSkillTagsModal}
      {governance.EditVersionMetadataModal}
      {governance.EditAliasesModal}
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.SKILL_REGISTRY, SkillDetailPage);
