import { Empty, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { RegistryVersionList } from '../../common/components/RegistryVersionList';
import Utils from '../../common/utils/Utils';
import type { SkillVersion } from '../types';
import { formatSkillStatusLabel, STATUS_TAG_COLOR } from '../utils';
import { flexColumnGapStyles, flexRowWrapStyles } from '../styles';

const SkillVersionSummary = ({ version }: { version: SkillVersion }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

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
const renderVersion = (version: SkillVersion) => <SkillVersionSummary version={version} />;

export const SkillVersionList = ({
  versions,
  selectedVersion,
  onSelectVersion,
  isLoading,
  hasMoreVersions,
}: {
  versions?: SkillVersion[];
  selectedVersion?: number;
  onSelectVersion: (version: number) => void;
  isLoading?: boolean;
  hasMoreVersions?: boolean;
}) => {
  const intl = useIntl();
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
