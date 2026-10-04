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
  NewWindowIcon,
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

const ExternalLink = ({ componentId, href }: { componentId: string; href: string }) => (
  <Typography.Link componentId={componentId} href={href} target="_blank" rel="noopener noreferrer">
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
          (source.locatorHref ? (
            <ExternalLink componentId="mlflow.skill_registry.detail.version.source" href={source.locatorHref} />
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
        <ExternalLink componentId="mlflow.skill_registry.detail.version.source_browse" href={source.browseHref} />
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
          renderDisplayedValue={(value) => formatSkillStatusLabel(value as SkillStatus)}
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
                {formatSkillStatusLabel(option)}
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
        {formatSkillStatusLabel(status)}
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

const DeleteVersionButton = ({
  status,
  isOnlyLiveVersion,
  onDelete,
}: {
  status: SkillStatus;
  isOnlyLiveVersion: boolean;
  onDelete: () => void;
}) => {
  const statusAllowsDelete = canSoftDeleteSkillVersion(status);
  const canDelete = statusAllowsDelete && !isOnlyLiveVersion;
  return (
    <Tooltip
      componentId="mlflow.skill_registry.detail.version.delete_tooltip"
      content={
        !statusAllowsDelete ? (
          <FormattedMessage
            defaultMessage="Unpublish or deprecate this version first. Deprecating keeps it resolving for anything that pins it."
            description="Tooltip when an active skill version cannot be deleted yet"
          />
        ) : isOnlyLiveVersion ? (
          <FormattedMessage
            defaultMessage="A skill's only remaining live version can't be deleted."
            description="Tooltip when the selected skill version is the last one that is not deleted"
          />
        ) : (
          <FormattedMessage
            defaultMessage="Removes this version from resolution, discovery and pull. Its number is never reused."
            description="Tooltip for deleting a draft or deprecated skill version"
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
  isOnlyLiveVersion = false,
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
  isOnlyLiveVersion?: boolean;
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
          {onDelete && (
            <DeleteVersionButton status={version.status} isOnlyLiveVersion={isOnlyLiveVersion} onDelete={onDelete} />
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

        <MetadataLabel>
          <FormattedMessage defaultMessage="Created by:" description="Skill version creator label" />
        </MetadataLabel>
        <MetadataValue>{version.created_by || '—'}</MetadataValue>

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
