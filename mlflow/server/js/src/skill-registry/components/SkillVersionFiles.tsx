import { useMemo, useState, type KeyboardEvent } from 'react';
import {
  Alert,
  ChevronDownIcon,
  ChevronRightIcon,
  FileIcon,
  FolderIcon,
  FolderOpenIcon,
  Modal,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { FormattedMessage, useIntl } from 'react-intl';

import { useSkillFileContentQuery, useSkillVersionFilesQuery } from '../hooks/useSkillVersionFiles';
import {
  buildSkillFileTree,
  formatFileSize,
  getPreviewLanguage,
  getSkillArtifactPath,
  looksBinary,
  MAX_LISTED_SKILL_FILES,
  SKILL_MANIFEST_FILE,
  type SkillFile,
  type SkillFileTreeNode,
} from '../skillFiles';
import type { SkillVersion } from '../types';
import { describeSkillSource } from '../utils';
import { SkillExternalLink } from './SkillExternalLink';

const onActivate = (action: () => void) => (event: KeyboardEvent) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    action();
  }
};

// The registry never fetches remote content, so only a Git version can point the user at its files.
const RemoteSourceNotice = ({ version }: { version: SkillVersion }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const source = describeSkillSource(version);
  const browseHref = version.source_type === 'git' ? (source.browseHref ?? source.locatorHref) : undefined;
  return (
    <Alert
      componentId="mlflow.skill_registry.detail.version.files.remote"
      type="info"
      closable={false}
      message={intl.formatMessage({
        defaultMessage: 'Content is read from a remote source.',
        description: 'Notice that a skill version stored outside MLflow has no file listing',
      })}
      description={
        browseHref && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, alignItems: 'flex-start' }}>
            <SkillExternalLink componentId="mlflow.skill_registry.detail.version.files.browse" href={browseHref} />
            <Typography.Text color="secondary" size="sm">
              <FormattedMessage
                defaultMessage="Opens a third-party site. Check you trust the source before using what it contains."
                description="Warning shown when a Skill version source links to an external site"
              />
            </Typography.Text>
          </div>
        )
      }
      css={{ maxWidth: 720 }}
    />
  );
};

const FileTree = ({
  nodes,
  activePath,
  onSelect,
}: {
  nodes: SkillFileTreeNode[];
  activePath?: string;
  onSelect: (file: SkillFile) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggle = (path: string) =>
    setCollapsed((current) => {
      const next = new Set(current);
      if (!next.delete(path)) next.add(path);
      return next;
    });
  const rowStyles = (depth: number) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.xs,
    paddingLeft: theme.spacing.sm + depth * theme.spacing.md,
    paddingRight: theme.spacing.sm,
    paddingBlock: 4,
    cursor: 'pointer',
    '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
  });
  const iconStyles = { flexShrink: 0, color: theme.colors.textSecondary };

  const renderNodes = (children: SkillFileTreeNode[], depth: number) =>
    children.map((node) => {
      if (!node.file) {
        const expanded = !collapsed.has(node.path);
        return (
          <div key={`dir:${node.path}`}>
            <div
              role="button"
              tabIndex={0}
              aria-expanded={expanded}
              onClick={() => toggle(node.path)}
              onKeyDown={onActivate(() => toggle(node.path))}
              css={rowStyles(depth)}
            >
              {expanded ? <ChevronDownIcon css={iconStyles} /> : <ChevronRightIcon css={iconStyles} />}
              {expanded ? <FolderOpenIcon css={iconStyles} /> : <FolderIcon css={iconStyles} />}
              <Typography.Text size="sm">{node.name}</Typography.Text>
            </div>
            {expanded && renderNodes(node.children, depth + 1)}
          </div>
        );
      }
      const file = node.file;
      return (
        <div
          key={`file:${node.path}`}
          role="button"
          tabIndex={0}
          onClick={() => onSelect(file)}
          onKeyDown={onActivate(() => onSelect(file))}
          css={{
            ...rowStyles(depth),
            // Files line up with folder names, past the chevron column.
            paddingLeft: theme.spacing.sm + depth * theme.spacing.md + theme.spacing.md,
            backgroundColor: activePath === file.path ? theme.colors.actionDefaultBackgroundPress : undefined,
          }}
        >
          <FileIcon css={iconStyles} />
          <Typography.Text
            size="sm"
            bold={file.path === SKILL_MANIFEST_FILE}
            css={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          >
            {node.name}
          </Typography.Text>
          <Typography.Text color="secondary" size="sm" css={{ flexShrink: 0 }}>
            {formatFileSize(file.size)}
          </Typography.Text>
        </div>
      );
    });

  return <>{renderNodes(nodes, 0)}</>;
};

const FilePreview = ({ artifactPath, file }: { artifactPath: string; file: SkillFile }) => {
  const { theme } = useDesignSystemTheme();
  const { data: content, isLoading, error, tooLarge } = useSkillFileContentQuery(artifactPath, file);
  if (tooLarge) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="This file is stored but too large to preview here. Pull the skill to read it."
          description="Skill file preview message for a large file"
        />
      </Typography.Text>
    );
  }
  if (isLoading) {
    return (
      <Typography.Hint>
        <FormattedMessage defaultMessage="Loading file..." description="Loading state for a skill file preview" />
      </Typography.Hint>
    );
  }
  if (error || content === undefined) {
    return (
      <Alert
        componentId="mlflow.skill_registry.detail.version.files.preview_error"
        type="error"
        closable={false}
        message={
          <FormattedMessage defaultMessage="Couldn't load this file." description="Skill file preview load error" />
        }
        description={error instanceof Error ? error.message : undefined}
      />
    );
  }
  if (looksBinary(content)) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="This file can't be previewed as text. Pull the skill to read it."
          description="Skill file preview message for a binary file"
        />
      </Typography.Text>
    );
  }
  return (
    <CodeSnippet
      language={getPreviewLanguage(file.path)}
      theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
      showLineNumbers
      wrapLongLines
      style={{
        padding: theme.spacing.sm,
        backgroundColor: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusMd,
        maxHeight: '60vh',
        overflow: 'auto',
        fontSize: theme.typography.fontSizeSm,
        lineHeight: theme.typography.lineHeightSm,
      }}
    >
      {content}
    </CodeSnippet>
  );
};

const StoredSkillFiles = ({ artifactPath }: { artifactPath: string }) => {
  const { theme } = useDesignSystemTheme();
  const { data, isLoading, error } = useSkillVersionFilesQuery(artifactPath);
  const [selected, setSelected] = useState<SkillFile>();
  const tree = useMemo(() => buildSkillFileTree(data?.files ?? []), [data]);

  if (isLoading) {
    return (
      <Typography.Hint>
        <FormattedMessage defaultMessage="Loading files..." description="Loading state for a skill version file list" />
      </Typography.Hint>
    );
  }
  if (error) {
    return (
      <Alert
        componentId="mlflow.skill_registry.detail.version.files.error"
        type="error"
        closable={false}
        message={
          <FormattedMessage
            defaultMessage="Couldn't load the files for this version."
            description="Skill version file list load error"
          />
        }
        description={error instanceof Error ? error.message : undefined}
        css={{ maxWidth: 720 }}
      />
    );
  }
  if (!tree.length) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="No files are stored for this version."
          description="Empty state for a skill version file list"
        />
      </Typography.Text>
    );
  }
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
      <div
        css={{
          minWidth: 0,
          maxHeight: 420,
          overflow: 'auto',
          paddingBlock: theme.spacing.xs,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
        }}
      >
        <FileTree nodes={tree} activePath={selected?.path} onSelect={setSelected} />
      </div>
      {data?.truncated && (
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Showing the first {count} files. Pull the skill to see the rest."
            description="Hint when a skill version has more files than the file list shows"
            values={{ count: MAX_LISTED_SKILL_FILES }}
          />
        </Typography.Hint>
      )}
      <Modal
        componentId="mlflow.skill_registry.detail.version.files.preview"
        title={selected?.path}
        visible={Boolean(selected)}
        onCancel={() => setSelected(undefined)}
        size="wide"
        footer={null}
      >
        {selected && <FilePreview artifactPath={artifactPath} file={selected} />}
      </Modal>
    </div>
  );
};

/** Files of a skill version: browsable when MLflow stores them, otherwise a pointer to the remote source. */
export const SkillVersionFiles = ({ version }: { version: SkillVersion }) => {
  const artifactPath = getSkillArtifactPath(version);
  // Keyed so switching versions drops an open preview, whose size check belongs to the old version's file.
  return artifactPath ? (
    <StoredSkillFiles key={artifactPath} artifactPath={artifactPath} />
  ) : (
    <RemoteSourceNotice version={version} />
  );
};
