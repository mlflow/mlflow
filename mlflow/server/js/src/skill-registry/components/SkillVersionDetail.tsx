import { useState, type ReactNode } from 'react';
import {
  Alert,
  Button,
  CopyIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  InfoSmallIcon,
  Spacer,
  Tag,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { SkillStatus, type Skill, type SkillVersion } from '../types';
import {
  aliasesForVersion,
  canSoftDeleteSkillVersion,
  describeSkillSource,
  formatSkillReferenceUris,
  formatSkillStatusLabel,
  isCommitSha,
  skillVersionStatusTransitions,
  STATUS_TAG_COLOR,
} from '../utils';
import { SkillAliases } from './SkillAliases';
import { SkillExternalLink } from './SkillExternalLink';
import { SkillVersionFiles } from './SkillVersionFiles';
import { SkillPencilButton } from './SkillPencilButton';
import { SkillTags } from './SkillTags';
import { UseSkillButton } from './UseSkillButton';
import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { flexRowStyles, inlineCodeStyles } from '../styles';
import Utils from '../../common/utils/Utils';

const EDITABLE_STATUS_ORDER = [SkillStatus.DRAFT, SkillStatus.ACTIVE, SkillStatus.DEPRECATED];

export interface SkillVersionStatusUpdate {
  onChange: (status: SkillStatus) => void;
  isPending: boolean;
  error?: Error | null;
  onDismissError: () => void;
}

const MetadataLabel = ({ children }: { children: ReactNode }) => <Typography.Text bold>{children}</Typography.Text>;

const MetadataValue = ({ children }: { children: ReactNode }) => (
  <Typography.Text css={{ minWidth: 0, overflowWrap: 'anywhere' }}>{children}</Typography.Text>
);

const InlineCode = ({ children }: { children: string }) => {
  const { theme } = useDesignSystemTheme();
  return <code css={inlineCodeStyles(theme)}>{children}</code>;
};

// Explains what the digest guarantees, since a missing one means pulls of the version are not checked.
const ContentDigestInfo = ({ version }: { version: SkillVersion }) => {
  const intl = useIntl();
  let content: ReactNode;
  if (version.digest) {
    content = (
      <FormattedMessage
        defaultMessage="SHA-256 of the skill's files, recorded at registration. Pulling this version fails if the fetched content no longer matches."
        description="Tooltip for a recorded skill version content digest"
      />
    );
  } else if (version.source_type === 'mlflow') {
    content = (
      <FormattedMessage
        defaultMessage="No digest was recorded. This version's files are stored in MLflow, so its content does not change."
        description="Tooltip for a missing digest on an uploaded skill version"
      />
    );
  } else if (version.source_type === 'git' && !isCommitSha(version.ref)) {
    content = version.ref ? (
      <FormattedMessage
        defaultMessage="No digest was recorded, so pulls aren't checked. This version points at {ref}; if that is a branch, a pull returns whatever it holds at the time."
        description="Tooltip for a missing digest on a skill version that points at a Git branch or tag"
        values={{ ref: version.ref }}
      />
    ) : (
      <FormattedMessage
        defaultMessage="No digest was recorded, so pulls aren't checked. This version points at the repository's default branch, so a pull returns whatever it holds at the time."
        description="Tooltip for a missing digest on a skill version that points at the default Git branch"
      />
    );
  } else {
    content = (
      <FormattedMessage
        defaultMessage="No digest was recorded, so pulls aren't checked against the content that was registered."
        description="Tooltip for a missing digest on a skill version"
      />
    );
  }
  return (
    <Tooltip componentId="mlflow.skill_registry.detail.version.digest_info" content={content} side="bottom">
      <InfoSmallIcon
        aria-label={intl.formatMessage({
          defaultMessage: 'About the content digest',
          description: 'Aria label for the skill version content digest info icon',
        })}
      />
    </Tooltip>
  );
};

const PaneMessage = ({ children, centered = false }: { children: ReactNode; centered?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        flex: 1,
        padding: centered ? theme.spacing.lg : theme.spacing.md,
        ...(centered ? { display: 'flex', alignItems: 'center', justifyContent: 'center' } : {}),
      }}
    >
      <Typography.Text color="secondary">{children}</Typography.Text>
    </div>
  );
};

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
          (source.locatorHref ? (
            <SkillExternalLink componentId="mlflow.skill_registry.detail.version.source" href={source.locatorHref} />
          ) : (
            <InlineCode>{source.locator}</InlineCode>
          ))}
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
      {source.browseHref && (
        <SkillExternalLink componentId="mlflow.skill_registry.detail.version.source_browse" href={source.browseHref} />
      )}
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

/** Rendered with a key per version and status, so a new selection or a saved change closes the editor. */
const SkillVersionStatusEditor = ({
  status,
  statusUpdate,
}: {
  status: SkillStatus;
  statusUpdate?: SkillVersionStatusUpdate;
}) => {
  const intl = useIntl();
  const [editing, setEditing] = useState(false);
  const transitions = skillVersionStatusTransitions(status);
  const label = intl.formatMessage({
    defaultMessage: 'Version status',
    description: 'Aria label for changing a skill version status',
  });

  if (editing && statusUpdate) {
    const allowed = new Set([status, ...transitions]);
    return (
      <DialogCombobox
        id="mlflow.skill_registry.detail.version.status_select"
        componentId="mlflow.skill_registry.detail.version.status_select"
        label={label}
        value={[status]}
        open
      >
        <DialogComboboxTrigger
          aria-label={label}
          withInlineLabel={false}
          renderDisplayedValue={(value) => formatSkillStatusLabel(intl, value as SkillStatus)}
          allowClear={false}
          width={160}
        />
        <DialogComboboxContent
          matchTriggerWidth
          onEscapeKeyDown={() => setEditing(false)}
          onPointerDownOutside={() => setEditing(false)}
        >
          <DialogComboboxOptionList>
            {EDITABLE_STATUS_ORDER.filter((option) => allowed.has(option)).map((option) => (
              <DialogComboboxOptionListSelectItem
                key={option}
                value={option}
                checked={option === status}
                onChange={() => {
                  setEditing(false);
                  if (option !== status) statusUpdate.onChange(option);
                }}
              >
                {formatSkillStatusLabel(intl, option)}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
    );
  }

  return (
    <>
      <Tag componentId="mlflow.skill_registry.detail.version.status" color={STATUS_TAG_COLOR[status]}>
        {formatSkillStatusLabel(intl, status)}
      </Tag>
      {statusUpdate && transitions.length > 0 && (
        <SkillPencilButton
          componentId="mlflow.skill_registry.detail.version.status_edit"
          label={intl.formatMessage({
            defaultMessage: 'Edit version status',
            description: 'Aria label for the skill version status pencil',
          })}
          disabled={statusUpdate.isPending}
          onClick={() => setEditing(true)}
        />
      )}
    </>
  );
};

const DeleteVersionButton = ({ status, onDelete }: { status: SkillStatus; onDelete: () => void }) => {
  const canDelete = canSoftDeleteSkillVersion(status);
  return (
    <Tooltip
      componentId="mlflow.skill_registry.detail.version.delete_tooltip"
      content={
        canDelete ? (
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
          danger={canDelete}
          type="primary"
          disabled={!canDelete}
          onClick={onDelete}
        >
          <FormattedMessage
            defaultMessage="Delete version"
            description="Button that soft-deletes the selected skill version"
          />
        </Button>
      </span>
    </Tooltip>
  );
};

/** Edit callbacks are only passed when the user may perform them. */
export const SkillVersionDetail = ({
  skill,
  version,
  isLoading,
  isMissing,
  error,
  onEditAliases,
  onEditMetadata,
  onDelete,
  statusUpdate,
}: {
  skill: Skill;
  version?: SkillVersion;
  isLoading?: boolean;
  isMissing?: boolean;
  error?: Error | null;
  onEditAliases?: () => void;
  onEditMetadata?: () => void;
  onDelete?: () => void;
  statusUpdate?: SkillVersionStatusUpdate;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (isLoading) {
    return (
      <PaneMessage>
        <FormattedMessage defaultMessage="Loading version..." description="Loading state for Skill version detail" />
      </PaneMessage>
    );
  }
  if (error && !isMissing) {
    return (
      <PaneMessage>
        {error.message || (
          <FormattedMessage
            defaultMessage="Failed to load this version."
            description="Skill detail message when the selected version cannot be loaded"
          />
        )}
      </PaneMessage>
    );
  }
  if (isMissing) {
    return (
      <PaneMessage centered>
        <FormattedMessage
          defaultMessage="This version is no longer available."
          description="Skill detail message when the selected version is missing or deleted"
        />
      </PaneMessage>
    );
  }
  if (!version) {
    return (
      <PaneMessage centered>
        <FormattedMessage
          defaultMessage="Select a version to view details."
          description="Skill detail placeholder when no version is selected"
        />
      </PaneMessage>
    );
  }

  const aliases = aliasesForVersion(skill, version);
  const referenceUris = formatSkillReferenceUris(skill.name, skill.organization, version.version, aliases);

  return (
    <div css={{ flex: 1, padding: theme.spacing.md, overflow: 'auto' }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: theme.spacing.sm }}>
        <Typography.Title level={3} withoutMargins>
          <FormattedMessage
            defaultMessage="Viewing version {version}"
            description="Skill version detail heading"
            values={{ version: version.version }}
          />
        </Typography.Title>
        <span css={{ display: 'inline-flex', gap: theme.spacing.sm }}>
          {onDelete && <DeleteVersionButton status={version.status} onDelete={onDelete} />}
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
      {statusUpdate?.error && (
        <Alert
          componentId="mlflow.skill_registry.detail.version.status_update_error"
          type="error"
          closable
          onClose={statusUpdate.onDismissError}
          message={statusUpdate.error.message}
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}
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
          <SkillVersionStatusEditor
            key={`${version.version}:${version.status}`}
            status={version.status}
            statusUpdate={statusUpdate}
          />
        </span>

        {/* Without auth the server records no user, so an empty creator is hidden rather than shown as "—". */}
        {version.created_by && (
          <>
            <MetadataLabel>
              <FormattedMessage defaultMessage="Created by:" description="Skill version creator label" />
            </MetadataLabel>
            <MetadataValue>{version.created_by}</MetadataValue>
          </>
        )}

        <MetadataLabel>
          <FormattedMessage defaultMessage="Aliases:" description="Skill version aliases label" />
        </MetadataLabel>
        <SkillAliases aliases={aliases} onEdit={onEditAliases} />

        <MetadataLabel>
          <FormattedMessage defaultMessage="Metadata:" description="Skill version tags label" />
        </MetadataLabel>
        <span css={flexRowStyles(theme)}>
          <SkillTags tags={version.tags || {}} wrap />
          {onEditMetadata && (
            <SkillPencilButton
              componentId="mlflow.skill_registry.detail.version.metadata.edit"
              label={intl.formatMessage({
                defaultMessage: 'Edit version tags',
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
          <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <FormattedMessage defaultMessage="Content digest:" description="Skill version content digest label" />
            <ContentDigestInfo version={version} />
          </span>
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

        <MetadataLabel>
          <FormattedMessage defaultMessage="Files:" description="Skill version files label" />
        </MetadataLabel>
        <SkillVersionFiles version={version} />
      </div>
    </div>
  );
};
