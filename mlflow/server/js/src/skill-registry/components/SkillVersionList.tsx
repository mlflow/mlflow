import { useCallback } from 'react';
import { Empty, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { AliasTag } from '../../common/components/AliasTag';
import { RegistryVersionList } from '../../common/components/RegistryVersionList';
import Utils from '../../common/utils/Utils';
import type { SkillVersion } from '../types';
import { formatSkillStatusLabel, STATUS_TAG_COLOR } from '../utils';
import { flexColumnGapStyles, flexRowWrapStyles, tagListStyles } from '../styles';
import { SkillLatestAliasTag } from './SkillLatestAliasTag';

const SkillVersionSummary = ({ version, isLatest }: { version: SkillVersion; isLatest: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const aliases = (version.aliases ?? []).filter((alias) => alias !== 'latest');

  return (
    <div css={flexColumnGapStyles(theme)}>
      <div css={flexRowWrapStyles(theme)}>
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Version {version}"
            description="Skill version list item label"
            values={{ version: version.version }}
          />
        </Typography.Text>
        <Tag componentId="mlflow.skill_registry.detail.version_status_tag" color={STATUS_TAG_COLOR[version.status]}>
          {formatSkillStatusLabel(intl, version.status)}
        </Tag>
        {(isLatest || aliases.length > 0) && (
          <div css={tagListStyles(theme)}>
            {isLatest && <SkillLatestAliasTag />}
            {aliases.map((alias) => (
              <AliasTag value={alias} key={alias} />
            ))}
          </div>
        )}
      </div>
      {version.creation_timestamp && (
        <Typography.Text size="sm" color="secondary">
          {Utils.formatTimestamp(version.creation_timestamp, intl)}
        </Typography.Text>
      )}
    </div>
  );
};

const getVersionKey = (version: SkillVersion) => String(version.version);

export const SkillVersionList = ({
  versions,
  latestVersion,
  selectedVersion,
  onSelectVersion,
  isLoading,
  hasMoreVersions,
}: {
  versions?: SkillVersion[];
  latestVersion?: number | null;
  selectedVersion?: number;
  onSelectVersion: (version: number) => void;
  isLoading?: boolean;
  hasMoreVersions?: boolean;
}) => {
  const intl = useIntl();
  const renderVersion = useCallback(
    (version: SkillVersion) => (
      <SkillVersionSummary version={version} isLatest={latestVersion != null && version.version === latestVersion} />
    ),
    [latestVersion],
  );
  return (
    <RegistryVersionList
      versions={versions}
      getVersionKey={getVersionKey}
      renderVersion={renderVersion}
      selectedKey={selectedVersion == null ? undefined : String(selectedVersion)}
      onSelect={(key) => onSelectVersion(Number(key))}
      header={intl.formatMessage({
        defaultMessage: 'Versions',
        description: 'Header for the version column in the Skill versions table',
      })}
      componentId="mlflow.skill_registry.detail.versions"
      emptyState={
        <Empty
          title={
            <FormattedMessage defaultMessage="No versions" description="Empty state when a Skill has no versions" />
          }
          // The list leaves out deleted versions, so it can't tell a new skill from one whose versions were deleted.
          description={null}
        />
      }
      isLoading={isLoading}
      hasMoreVersions={hasMoreVersions}
    />
  );
};
