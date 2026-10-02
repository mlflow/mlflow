import { useEffect } from 'react';
import {
  Alert,
  Breadcrumb,
  GenericSkeleton,
  Header,
  LockIcon,
  Spacer,
  TableSkeleton,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { Link, useParams } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import SkillRegistryRoutes from '../routes';
import {
  formatSkillOrganization,
  isNotFoundError,
  isPermissionDeniedError,
  isSkillDimmed,
  parseSkillRouteParams,
  resolveDefaultSkillVersion,
} from '../utils';
import { headerIconStyles, textClampStyles } from '../styles';
import { useSkillQuery } from '../hooks/useSkillQuery';
import { useSkillVersionQuery, useSkillVersionsQuery } from '../hooks/useSkillVersionsQuery';
import { useSelectedSkillVersion } from '../hooks/useSelectedSkillVersion';
import { SkillIcon } from '../components/SkillIcon';
import { SkillTags } from '../components/SkillTags';
import { SkillVersionList } from '../components/SkillVersionList';
import { SkillVersionDetail } from '../components/SkillVersionDetail';
import { SkillRegistryEmptyState } from '../components/SkillRegistryEmptyState';
import type { Skill } from '../types';
import { SkillStatus } from '../types';

const breadcrumbs = (
  <Breadcrumb>
    <Breadcrumb.Item>
      <Link componentId="mlflow.skill_registry.detail.breadcrumb_back" to={SkillRegistryRoutes.skillRegistryPageRoute}>
        <FormattedMessage defaultMessage="Skills" description="Breadcrumb link back to the Skill Registry catalog" />
      </Link>
    </Breadcrumb.Item>
  </Breadcrumb>
);

const SkillDetailHeader = ({ skill }: { skill: Skill }) => {
  const { theme } = useDesignSystemTheme();
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
      />
      {organizationLabel && (
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
          {organizationLabel}
        </Typography.Text>
      )}
      {skill.description && (
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs, ...textClampStyles(3) }}>
          {skill.description}
        </Typography.Text>
      )}
      {hasTags && (
        <div css={{ marginTop: theme.spacing.xs }}>
          <SkillTags tags={skill.tags} wrap />
        </div>
      )}
    </>
  );
};

const SkillDetailPage = () => {
  const { theme } = useDesignSystemTheme();
  const params = useParams<{ skillKey?: string; organization?: string; skillName?: string }>();
  const { name, organization } = parseSkillRouteParams(params);
  const [selectedVersion, setSelectedVersion] = useSelectedSkillVersion();
  const { data: skill, isLoading: skillLoading, error: skillError, refetch } = useSkillQuery(name, organization);
  const {
    data: versions,
    isLoading: versionsLoading,
    error: versionsError,
  } = useSkillVersionsQuery(name, organization);

  const selectedFromList = versions?.find((version) => version.version === selectedVersion);
  const shouldFetchSelectedVersion = selectedVersion != null && !selectedFromList && !versionsLoading;
  const {
    data: fetchedVersion,
    isLoading: fetchedVersionLoading,
    error: fetchedVersionError,
  } = useSkillVersionQuery(name, organization, selectedVersion, shouldFetchSelectedVersion);
  const currentVersion = selectedFromList ?? fetchedVersion;
  const selectedVersionMissing =
    shouldFetchSelectedVersion &&
    !fetchedVersionLoading &&
    (isNotFoundError(fetchedVersionError) || currentVersion?.status === SkillStatus.DELETED);
  const selectedVersionError =
    shouldFetchSelectedVersion && !fetchedVersionLoading && fetchedVersionError && !selectedVersionMissing
      ? fetchedVersionError
      : undefined;
  const isVersionDetailLoading =
    Boolean(selectedVersion) &&
    !currentVersion &&
    !selectedVersionMissing &&
    !selectedVersionError &&
    (versionsLoading || fetchedVersionLoading);

  useEffect(() => {
    if (selectedVersion != null || skillLoading || !skill) {
      return;
    }
    if (skill.latest_version != null) {
      setSelectedVersion(skill.latest_version);
      return;
    }
    if (versionsLoading) {
      return;
    }
    const next = resolveDefaultSkillVersion(skill, versions);
    if (next != null) {
      setSelectedVersion(next);
    }
  }, [selectedVersion, skill, skillLoading, versions, versionsLoading, setSelectedVersion]);

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
      <SkillDetailHeader skill={skill} />
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
            version={selectedVersionMissing ? undefined : currentVersion}
            isLoading={isVersionDetailLoading}
            isMissing={selectedVersionMissing}
            error={selectedVersionError}
          />
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.SKILL_REGISTRY, SkillDetailPage);
