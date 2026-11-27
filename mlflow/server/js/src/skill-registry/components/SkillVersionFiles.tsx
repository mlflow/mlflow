import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import {
  Alert,
  Button,
  ChevronDownIcon,
  CopyIcon,
  ChevronRightIcon,
  FileIcon,
  FolderIcon,
  FolderOpenIcon,
  Modal,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { FormattedMessage, useIntl } from 'react-intl';

import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { GenAIMarkdownRenderer } from '../../shared/web-shared/genai-markdown-renderer';
import { sanitizeHref } from '../../common/utils/registryIcons';
import { useSkillFileContentQuery, useSkillVersionFilesQuery } from '../hooks/useSkillVersionFiles';
import { readFrontmatterFields, readSkillManifest, splitFrontmatter } from '../localSkillFolder';
import {
  buildSkillFileTree,
  formatFileSize,
  getCodeBlockLanguage,
  getPreviewLanguage,
  getSkillArtifactPath,
  looksBinary,
  SKILL_MANIFEST_FILE,
  type SkillFile,
  type SkillFileTreeNode,
} from '../skillFiles';
import type { SkillVersion } from '../types';
import { SkillSectionCard } from './SkillSectionCard';

const onActivate = (action: () => void) => (event: KeyboardEvent) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    action();
  }
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

const isMarkdownFile = (path: string) => /\.(md|markdown)$/i.test(path);

const PreviewFrame = ({ children }: { children: ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        maxHeight: '60vh',
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        // The frame scrolls; its children keep their full height instead of shrinking to fit it.
        '& > *': { flexShrink: 0 },
      }}
    >
      {children}
    </div>
  );
};

const TextSnippet = ({ language, children }: { language: ReturnType<typeof getPreviewLanguage>; children: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <CodeSnippet
      language={language}
      theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
      showLineNumbers
      wrapLongLines
      style={{
        padding: theme.spacing.sm,
        paddingRight: theme.spacing.xl + theme.spacing.sm,
        backgroundColor: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusMd,
        fontSize: theme.typography.fontSizeSm,
        lineHeight: theme.typography.lineHeightSm,
        // Long lines wrap and the frame around the snippet scrolls, so the snippet itself never does.
        overflow: 'visible',
      }}
    >
      {children}
    </CodeSnippet>
  );
};

const WithCopyButton = ({
  componentId,
  label,
  text,
  className,
  children,
}: {
  componentId: string;
  label: string;
  text: string;
  className?: string;
  children: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ position: 'relative' }} className={className}>
      <CopyButton
        componentId={componentId}
        showLabel={false}
        copyText={text}
        icon={<CopyIcon />}
        aria-label={label}
        css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.md, zIndex: 1 }}
      />
      {children}
    </div>
  );
};

/** A file's text as is, with a copy button that stays in place while the text scrolls. */
const RawFile = ({ path, content }: { path: string; content: string }) => {
  const intl = useIntl();
  return (
    <WithCopyButton
      componentId="mlflow.skill_registry.detail.version.files.copy"
      label={intl.formatMessage({
        defaultMessage: 'Copy file contents',
        description: 'Aria label for copying a skill file preview',
      })}
      text={content}
    >
      <PreviewFrame>
        <TextSnippet language={getPreviewLanguage(path)}>{content}</TextSnippet>
      </PreviewFrame>
    </WithCopyButton>
  );
};

// Code blocks wrap rather than scroll, so the rendered file has a single scrollbar: the frame's.
const MarkdownCodeBlock = ({ children, language }: { children?: ReactNode; language?: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const code = String(children).replace(/\n$/, '');
  return (
    <WithCopyButton
      componentId="mlflow.skill_registry.detail.version.files.copy_code"
      label={intl.formatMessage({
        defaultMessage: 'Copy code',
        description: 'Aria label for copying a code block in a rendered skill markdown file',
      })}
      text={code}
      css={{ marginBottom: theme.spacing.md }}
    >
      <TextSnippet language={getCodeBlockLanguage(language)}>{code}</TextSnippet>
    </WithCopyButton>
  );
};

// Images in a skill's markdown are shown as links rather than loaded, so opening a file never fetches a remote
// URL; relative ones point into the skill and are shown as text.
const markdownComponents = {
  img: ({ src, alt }: { src?: string; alt?: string }) => {
    const href = sanitizeHref(src);
    const label = alt || src;
    return href ? (
      <Typography.Link componentId="mlflow.skill_registry.detail.version.files.image_link" href={href} openInNewTab>
        {label}
      </Typography.Link>
    ) : (
      <Typography.Text color="secondary">{label}</Typography.Text>
    );
  },
  codeBlock: MarkdownCodeBlock,
  // The code block brings its own <pre>; the one markdown wraps around it would scroll on its own.
  pre: ({ children }: { children?: ReactNode }) => <>{children}</>,
};

// Frontmatter is shown as a table of its fields, as code hosts do; text that is not a YAML mapping stays as is.
const Frontmatter = ({ source }: { source: string }) => {
  const { theme } = useDesignSystemTheme();
  const fields = readFrontmatterFields(source);
  if (!fields) return <TextSnippet language="text">{source}</TextSnippet>;
  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: 'max-content 1fr',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        '& > *': { padding: `${theme.spacing.xs}px ${theme.spacing.sm}px` },
        '& > :nth-of-type(n + 3)': { borderTop: `1px solid ${theme.colors.border}` },
      }}
    >
      {fields.map(([key, value]) => [
        <Typography.Text key={`key:${key}`} bold>
          {key}
        </Typography.Text>,
        <Typography.Text key={`value:${key}`} css={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
          {typeof value === 'string' ? value : JSON.stringify(value)}
        </Typography.Text>,
      ])}
    </div>
  );
};

/** A markdown file, such as SKILL.md: raw text by default, or rendered with its frontmatter as a table above it. */
const MarkdownFile = ({ path, content }: { path: string; content: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [view, setView] = useState<'preview' | 'raw'>('raw');
  const { frontmatter, body } = splitFrontmatter(content);
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <SegmentedControlGroup
        name="mlflow.skill_registry.detail.version.files.view"
        componentId="mlflow.skill_registry.detail.version.files.view"
        value={view}
        onChange={(event) => setView(event.target.value as 'preview' | 'raw')}
        aria-label={intl.formatMessage({
          defaultMessage: 'File view',
          description: 'Aria label for switching a markdown file between rendered and raw text',
        })}
        css={{ alignSelf: 'flex-start' }}
      >
        <SegmentedControlButton value="raw">
          <FormattedMessage defaultMessage="Raw" description="Show a markdown file as raw text" />
        </SegmentedControlButton>
        <SegmentedControlButton value="preview">
          <FormattedMessage defaultMessage="Preview" description="Show a markdown file rendered" />
        </SegmentedControlButton>
      </SegmentedControlGroup>
      {view === 'raw' ? (
        <RawFile path={path} content={content} />
      ) : (
        <PreviewFrame>
          {frontmatter && <Frontmatter source={frontmatter} />}
          <GenAIMarkdownRenderer components={markdownComponents}>{body}</GenAIMarkdownRenderer>
        </PreviewFrame>
      )}
    </div>
  );
};

const FilePreview = ({ artifactPath, file }: { artifactPath: string; file: SkillFile }) => {
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
  return isMarkdownFile(file.path) ? (
    <MarkdownFile path={file.path} content={content} />
  ) : (
    <RawFile path={file.path} content={content} />
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
            values={{ count: data.files.length }}
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

const COLLAPSED_MANIFEST_HEIGHT = 280;

const readManifestFields = (content: string) => {
  try {
    return readSkillManifest(content);
  } catch {
    return {};
  }
};

const ManifestField = ({ label, children }: { label: ReactNode; children: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text color="secondary">{label}</Typography.Text>
      <Typography.Text css={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{children}</Typography.Text>
    </div>
  );
};

// The name and description an agent reads to decide when to load the skill, shown above the instructions.
const ManifestFields = ({ content }: { content: string }) => {
  const { theme } = useDesignSystemTheme();
  const { name, description } = readManifestFields(content);
  if (!name && !description) return null;
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        paddingBottom: theme.spacing.md,
        borderBottom: `1px solid ${theme.colors.border}`,
      }}
    >
      {name && (
        <ManifestField label={<FormattedMessage defaultMessage="Skill name" description="SKILL.md name field label" />}>
          {name}
        </ManifestField>
      )}
      {description && (
        <ManifestField
          label={<FormattedMessage defaultMessage="Description" description="SKILL.md description field label" />}
        >
          {description}
        </ManifestField>
      )}
    </div>
  );
};

const ManifestBody = ({ content }: { content: string }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const { body } = splitFrontmatter(content);

  // Whether the rendered markdown is taller than the collapsed height can only be measured once it is on screen,
  // and changes as the page's width rewraps it.
  useEffect(() => {
    const element = bodyRef.current;
    if (!element) return undefined;
    const measure = () => setOverflows(element.scrollHeight > COLLAPSED_MANIFEST_HEIGHT);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [body]);

  const collapsed = overflows && !expanded;
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <ManifestFields content={content} />
      <div
        ref={bodyRef}
        // Links and copy buttons below the fold stay in the tab order, so tabbing into the preview reveals the rest.
        // Tab's keyup lands on the newly focused element; a click on a visible link focuses it without expanding.
        onKeyUp={(event) => {
          if (collapsed && event.key === 'Tab') setExpanded(true);
        }}
        css={{
          position: 'relative',
          maxHeight: collapsed ? COLLAPSED_MANIFEST_HEIGHT : undefined,
          overflow: 'hidden',
          '&::after': collapsed
            ? {
                content: '""',
                position: 'absolute',
                insetInline: 0,
                bottom: 0,
                height: theme.spacing.lg * 2,
                background: `linear-gradient(transparent, ${theme.colors.backgroundPrimary})`,
                pointerEvents: 'none',
              }
            : undefined,
        }}
      >
        <GenAIMarkdownRenderer components={markdownComponents}>{body}</GenAIMarkdownRenderer>
      </div>
      {overflows && (
        <Button
          componentId="mlflow.skill_registry.detail.manifest.toggle"
          type="link"
          onClick={() => setExpanded((current) => !current)}
          css={{ alignSelf: 'flex-start' }}
        >
          {expanded ? (
            <FormattedMessage defaultMessage="Show less" description="Collapse the SKILL.md preview on a skill page" />
          ) : (
            <FormattedMessage defaultMessage="Show more" description="Expand the SKILL.md preview on a skill page" />
          )}
        </Button>
      )}
    </div>
  );
};

const StoredManifestPreview = ({ artifactPath }: { artifactPath: string }) => {
  const intl = useIntl();
  const { data } = useSkillVersionFilesQuery(artifactPath);
  const manifest = data?.files.find((file) => file.path === SKILL_MANIFEST_FILE);
  const { data: content, isLoading, error, tooLarge } = useSkillFileContentQuery(artifactPath, manifest);
  // Anything that keeps SKILL.md from loading is reported in the file list, which also offers the raw file.
  if (!manifest || tooLarge || error || (!isLoading && (content === undefined || looksBinary(content)))) return null;
  return (
    <SkillSectionCard
      title={SKILL_MANIFEST_FILE}
      actions={
        content !== undefined && (
          <CopyButton
            componentId="mlflow.skill_registry.detail.manifest.copy"
            showLabel={false}
            copyText={content}
            icon={<CopyIcon />}
            size="small"
            type="tertiary"
            aria-label={intl.formatMessage({
              defaultMessage: 'Copy SKILL.md',
              description: 'Aria label for copying the SKILL.md preview on a skill page',
            })}
          />
        )
      }
    >
      {isLoading || content === undefined ? (
        <Typography.Hint>
          <FormattedMessage defaultMessage="Loading SKILL.md..." description="Loading state for the SKILL.md preview" />
        </Typography.Hint>
      ) : (
        <ManifestBody content={content} />
      )}
    </SkillSectionCard>
  );
};

/** The opening of a stored version's SKILL.md, rendered, so the page shows what the skill does without a click. */
export const SkillManifestPreview = ({ version }: { version: SkillVersion }) => {
  const artifactPath = getSkillArtifactPath(version);
  // The registry never fetches remote content, so only versions MLflow stores have a preview.
  return artifactPath ? <StoredManifestPreview key={artifactPath} artifactPath={artifactPath} /> : null;
};

/** The files of a version MLflow stores. Remote versions have none to list; their source is under Details. */
export const SkillVersionFiles = ({ version }: { version: SkillVersion }) => {
  const artifactPath = getSkillArtifactPath(version);
  if (!artifactPath) return null;
  return (
    <SkillSectionCard
      title={<FormattedMessage defaultMessage="Files" description="Title of a skill version's files section" />}
    >
      {/* Keyed so switching versions drops an open preview, whose size check belongs to the old version's file. */}
      <StoredSkillFiles key={artifactPath} artifactPath={artifactPath} />
    </SkillSectionCard>
  );
};
