import type { ReactNode } from 'react';
import { CopyIcon, NewWindowIcon, Spacer, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { Skill, SkillVersion } from '../types';
import { aliasesForVersion, describeSkillSource, formatSkillReferenceUris, STATUS_TAG_COLOR } from '../utils';
import { SkillAliases } from './SkillAliases';
import { SkillTags } from './SkillTags';
import { UseSkillButton } from './UseSkillButton';
import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { flexRowStyles, inlineCodeStyles } from '../styles';
import Utils from '../../common/utils/Utils';

const MetadataLabel = ({ children }: { children: ReactNode }) => <Typography.Text bold>{children}</Typography.Text>;

const MetadataValue = ({ children }: { children: ReactNode }) => (
  <Typography.Text css={{ minWidth: 0, overflowWrap: 'anywhere' }}>{children}</Typography.Text>
);

const InlineCode = ({ children }: { children: string }) => {
  const { theme } = useDesignSystemTheme();
  return <code css={inlineCodeStyles(theme)}>{children}</code>;
};

const SourceLink = ({ href }: { href: string }) => (
  <Typography.Link
    componentId="mlflow.skill_registry.detail.version.source"
    href={href}
    target="_blank"
    rel="noopener noreferrer"
  >
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      {href}
      <span aria-hidden>
        <NewWindowIcon css={{ fontSize: 12 }} />
      </span>
    </span>
  </Typography.Link>
);

const BrowseLink = ({ href }: { href: string }) => (
  <Typography.Link
    componentId="mlflow.skill_registry.detail.version.source_browse"
    href={href}
    target="_blank"
    rel="noopener noreferrer"
  >
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      {href}
      <span aria-hidden>
        <NewWindowIcon css={{ fontSize: 12 }} />
      </span>
    </span>
  </Typography.Link>
);

const SkillSourceDetails = ({ version }: { version: SkillVersion }) => {
  const { theme } = useDesignSystemTheme();
  const source = describeSkillSource(version);
  if (!source.label && !source.locator) {
    return <MetadataValue>—</MetadataValue>;
  }

  return (
    <div
      css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs, minWidth: 0 }}
    >
      <span css={{ ...flexRowStyles(theme), flexWrap: 'wrap', minWidth: 0 }}>
        {source.label && (
          <Tag componentId="mlflow.skill_registry.detail.version.source_type" color="charcoal">
            {source.label}
          </Tag>
        )}
        {source.locator &&
          (source.locatorHref ? <SourceLink href={source.locatorHref} /> : <InlineCode>{source.locator}</InlineCode>)}
      </span>
      {source.path && (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Path: {path}"
            description="Skill version source subpath"
            values={{ path: source.path }}
          />
        </Typography.Text>
      )}
      {source.ref && (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Ref: {ref}"
            description="Skill version git ref"
            values={{ ref: source.ref }}
          />
        </Typography.Text>
      )}
      {source.browseHref && <BrowseLink href={source.browseHref} />}
      {source.showExternalWarning && (
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="Opens a third-party site. Check you trust the source before using what it contains."
            description="Warning shown when a Skill version source links to an external site"
          />
        </Typography.Text>
      )}
    </div>
  );
};

export const SkillVersionDetail = ({
  skill,
  version,
  isLoading,
  isMissing,
  error,
}: {
  skill: Skill;
  version?: SkillVersion;
  isLoading?: boolean;
  isMissing?: boolean;
  error?: Error | null;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (isLoading) {
    return (
      <div css={{ flex: 1, padding: theme.spacing.md }}>
        <Typography.Hint>
          <FormattedMessage defaultMessage="Loading version..." description="Loading state for Skill version detail" />
        </Typography.Hint>
      </div>
    );
  }

  if (error && !isMissing) {
    return (
      <div css={{ flex: 1, padding: theme.spacing.md }}>
        <Typography.Text color="secondary">
          {error.message || (
            <FormattedMessage
              defaultMessage="Failed to load this version."
              description="Skill detail message when the selected version cannot be loaded"
            />
          )}
        </Typography.Text>
      </div>
    );
  }

  if (isMissing) {
    return (
      <div
        css={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg,
        }}
      >
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="This version is no longer available."
            description="Skill detail message when the selected version is missing or deleted"
          />
        </Typography.Text>
      </div>
    );
  }

  if (!version) {
    return (
      <div
        css={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg,
        }}
      >
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Select a version to view details."
            description="Skill detail placeholder when no version is selected"
          />
        </Typography.Text>
      </div>
    );
  }

  const aliases = aliasesForVersion(skill, version);
  const referenceUris = formatSkillReferenceUris(skill.name, skill.organization, version.version, aliases);

  return (
    <div
      css={{
        flex: 1,
        padding: theme.spacing.md,
        overflow: 'auto',
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: theme.spacing.sm }}>
        <Typography.Title level={3} withoutMargins>
          <FormattedMessage
            defaultMessage="Viewing version {version}"
            description="Skill version detail heading"
            values={{ version: version.version }}
          />
        </Typography.Title>
        <UseSkillButton
          skill={skill}
          version={version.version}
          versionStatus={version.status}
          showLabel
          appearance="default"
        />
      </div>

      <Spacer shrinks={false} />
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '140px 1fr',
          gridAutoRows: `minmax(${theme.typography.lineHeightLg}, auto)`,
          alignItems: 'flex-start',
          rowGap: theme.spacing.xs,
          columnGap: theme.spacing.sm,
        }}
      >
        <MetadataLabel>
          <FormattedMessage defaultMessage="Registered at:" description="Skill version creation timestamp label" />
        </MetadataLabel>
        <MetadataValue>
          {version.creation_timestamp ? Utils.formatTimestamp(version.creation_timestamp) : '—'}
        </MetadataValue>

        <MetadataLabel>
          <FormattedMessage defaultMessage="Status:" description="Skill version stored status label" />
        </MetadataLabel>
        <span css={flexRowStyles(theme)}>
          <Tag componentId="mlflow.skill_registry.detail.version.status" color={STATUS_TAG_COLOR[version.status]}>
            {version.status}
          </Tag>
        </span>

        <MetadataLabel>
          <FormattedMessage defaultMessage="Created by:" description="Skill version creator label" />
        </MetadataLabel>
        <MetadataValue>{version.created_by || '—'}</MetadataValue>

        <MetadataLabel>
          <FormattedMessage defaultMessage="Aliases:" description="Skill version aliases label" />
        </MetadataLabel>
        <SkillAliases aliases={aliases} />

        <MetadataLabel>
          <FormattedMessage defaultMessage="Metadata:" description="Skill version tags label" />
        </MetadataLabel>
        <SkillTags tags={version.tags || {}} wrap />

        <MetadataLabel>
          <FormattedMessage defaultMessage="Source:" description="Skill version source label" />
        </MetadataLabel>
        <SkillSourceDetails version={version} />

        <MetadataLabel>
          <FormattedMessage defaultMessage="Content digest:" description="Skill version content digest label" />
        </MetadataLabel>
        <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, maxWidth: '100%' }}>
          {version.digest ? <InlineCode>{version.digest}</InlineCode> : <MetadataValue>—</MetadataValue>}
          {version.digest && (
            <CopyButton
              componentId="mlflow.skill_registry.detail.version.digest.copy"
              copyText={version.digest}
              showLabel={false}
              size="small"
              type="tertiary"
              icon={<CopyIcon css={{ fontSize: theme.typography.fontSizeSm }} />}
              aria-label={intl.formatMessage({
                defaultMessage: 'Copy content digest',
                description: 'Aria label for copying a Skill version content digest',
              })}
            />
          )}
        </span>

        <MetadataLabel>
          <FormattedMessage
            defaultMessage="Reference URIs:"
            description="Pinned skills URI label for a Skill version"
          />
        </MetadataLabel>
        <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs }}>
          {referenceUris.map((referenceUri) => (
            <InlineCode key={referenceUri}>{referenceUri}</InlineCode>
          ))}
        </div>
      </div>
    </div>
  );
};
