import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  Button,
  CopyIcon,
  NewWindowIcon,
  SimpleSelect,
  SimpleSelectOption,
  Spacer,
  Tag,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { Skill, SkillStatus, SkillVersion } from '../types';
import {
  aliasesForVersion,
  canSoftDeleteSkillVersion,
  describeSkillSource,
  formatSkillReferenceUris,
  skillVersionStatusTransitions,
  STATUS_TAG_COLOR,
} from '../utils';
import { SkillAliases } from './SkillAliases';
import { SkillPencilButton } from './SkillPencilButton';
import { SkillTags } from './SkillTags';
import { UseSkillButton } from './UseSkillButton';
import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { flexRowStyles, inlineCodeStyles } from '../styles';
import Utils from '../../common/utils/Utils';

const statusLabel = (status: SkillStatus) => status.charAt(0).toUpperCase() + status.slice(1);

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
  canUpdate = false,
  onEditAliases,
  onEditMetadata,
  onDelete,
  onStatusChange,
}: {
  skill: Skill;
  version?: SkillVersion;
  isLoading?: boolean;
  isMissing?: boolean;
  error?: Error | null;
  canUpdate?: boolean;
  onEditAliases?: () => void;
  onEditMetadata?: () => void;
  onDelete?: () => void;
  onStatusChange?: (status: SkillStatus) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [editingStatus, setEditingStatus] = useState(false);
  const statusSelectRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    setEditingStatus(false);
  }, [version?.version, version?.status]);
  useEffect(() => {
    if (!editingStatus) return;
    const closeOnOutside = (event: PointerEvent) => {
      if (statusSelectRef.current?.contains(event.target as Node)) return;
      setEditingStatus(false);
    };
    document.addEventListener('pointerdown', closeOnOutside);
    return () => document.removeEventListener('pointerdown', closeOnOutside);
  }, [editingStatus]);

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
        <span css={{ display: 'inline-flex', gap: theme.spacing.sm }}>
          {onDelete && (
            <Tooltip
              componentId="mlflow.skill_registry.detail.version.delete_tooltip"
              content={
                canSoftDeleteSkillVersion(version.status) ? (
                  <FormattedMessage
                    defaultMessage="Removes this version from resolution, discovery and pull. Its number is never reused."
                    description="Tooltip for deleting a draft or deprecated skill version"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Unpublish or deprecate this version first. Deprecating keeps it resolving for anything that pins it."
                    description="Tooltip when an active skill version cannot be deleted yet"
                  />
                )
              }
            >
              <span>
                <Button
                  componentId="mlflow.skill_registry.detail.version.delete"
                  icon={<TrashIcon />}
                  danger={canSoftDeleteSkillVersion(version.status)}
                  type="primary"
                  disabled={!canSoftDeleteSkillVersion(version.status)}
                  onClick={onDelete}
                >
                  <FormattedMessage
                    defaultMessage="Delete version"
                    description="Button that soft-deletes the selected skill version"
                  />
                </Button>
              </span>
            </Tooltip>
          )}
          <UseSkillButton
            skill={skill}
            version={version.version}
            versionStatus={version.status}
            showLabel
            appearance="default"
          />
        </span>
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
          {editingStatus && onStatusChange ? (
            <span ref={statusSelectRef}>
              <SimpleSelect
                id="mlflow.skill_registry.detail.version.status_select"
                componentId="mlflow.skill_registry.detail.version.status_select"
                aria-label={intl.formatMessage({
                  defaultMessage: 'Version status',
                  description: 'Aria label for changing a skill version status',
                })}
                value={version.status}
                onChange={({ target }) => {
                  const nextStatus = target.value as SkillStatus;
                  setEditingStatus(false);
                  if (nextStatus !== version.status) onStatusChange(nextStatus);
                }}
              >
                <SimpleSelectOption value={version.status}>{statusLabel(version.status)}</SimpleSelectOption>
                {skillVersionStatusTransitions(version.status).map((status) => (
                  <SimpleSelectOption key={status} value={status}>
                    {statusLabel(status)}
                  </SimpleSelectOption>
                ))}
              </SimpleSelect>
            </span>
          ) : (
            <>
              <Tag componentId="mlflow.skill_registry.detail.version.status" color={STATUS_TAG_COLOR[version.status]}>
                {statusLabel(version.status)}
              </Tag>
              {canUpdate && onStatusChange && skillVersionStatusTransitions(version.status).length > 0 && (
                <SkillPencilButton
                  componentId="mlflow.skill_registry.detail.version.status_edit"
                  label={intl.formatMessage({
                    defaultMessage: 'Edit status',
                    description: 'Aria label for the skill version status pencil',
                  })}
                  onClick={() => setEditingStatus(true)}
                />
              )}
            </>
          )}
        </span>

        <MetadataLabel>
          <FormattedMessage defaultMessage="Created by:" description="Skill version creator label" />
        </MetadataLabel>
        <MetadataValue>{version.created_by || '—'}</MetadataValue>

        <MetadataLabel>
          <FormattedMessage defaultMessage="Aliases:" description="Skill version aliases label" />
        </MetadataLabel>
        <SkillAliases aliases={aliases} onEdit={canUpdate ? onEditAliases : undefined} />

        <MetadataLabel>
          <FormattedMessage defaultMessage="Metadata:" description="Skill version tags label" />
        </MetadataLabel>
        <span css={flexRowStyles(theme)}>
          <SkillTags tags={version.tags || {}} wrap />
          {canUpdate && onEditMetadata && (
            <SkillPencilButton
              componentId="mlflow.skill_registry.detail.version.metadata.edit"
              label={intl.formatMessage({
                defaultMessage: 'Edit metadata',
                description: 'Aria label for editing skill version tags',
              })}
              onClick={onEditMetadata}
            />
          )}
        </span>

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
