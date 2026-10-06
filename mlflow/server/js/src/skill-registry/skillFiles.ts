import type { CodeSnippetLanguage } from '@databricks/web-shared/snippet';

import { SkillRegistryApi } from './api';
import type { SkillVersion } from './types';

const MLFLOW_ARTIFACTS_SCHEME = 'mlflow-artifacts:';
export const SKILL_MANIFEST_FILE = 'SKILL.md';
// Skills are small text trees (the server caps uploads at 25 MiB), so a cap only guards against odd content.
export const MAX_LISTED_SKILL_FILES = 500;
export const MAX_PREVIEW_BYTES = 512 * 1024;

export interface SkillFile {
  /** Path relative to the skill root, using forward slashes. */
  path: string;
  size?: number;
}

export interface SkillFileTreeNode {
  name: string;
  path: string;
  file?: SkillFile;
  children: SkillFileTreeNode[];
}

/**
 * Artifact path of a version whose content MLflow stores (`source_type="mlflow"`), relative to the
 * artifact root. Remote sources return undefined: the registry never fetches them.
 */
export const getSkillArtifactPath = (version: Pick<SkillVersion, 'source_type' | 'source' | 'subpath'>) => {
  if (version.source_type !== 'mlflow' || !version.source?.startsWith(MLFLOW_ARTIFACTS_SCHEME)) return undefined;
  const base = version.source.slice(MLFLOW_ARTIFACTS_SCHEME.length).replace(/^\/+|\/+$/g, '');
  const subpath = version.subpath?.replace(/^\/+|\/+$/g, '');
  return subpath ? `${base}/${subpath}` : base || undefined;
};

/** Walks the stored tree breadth first, one listing call per directory. */
export const listSkillFiles = async (rootPath: string) => {
  const files: SkillFile[] = [];
  const pending = [''];
  let truncated = false;
  while (pending.length) {
    const directory = pending.shift() as string;
    const { files: entries = [] } = await SkillRegistryApi.listArtifacts(
      directory ? `${rootPath}/${directory}` : rootPath,
    );
    for (const entry of entries) {
      const path = directory ? `${directory}/${entry.path}` : entry.path;
      if (entry.is_dir) {
        pending.push(path);
      } else if (files.length < MAX_LISTED_SKILL_FILES) {
        files.push({ path, size: entry.file_size });
      } else {
        truncated = true;
      }
    }
    if (truncated) break;
  }
  return { files, truncated };
};

const isManifest = (node: SkillFileTreeNode) => node.file?.path === SKILL_MANIFEST_FILE;

// SKILL.md first, then files before folders, then by name.
const compareNodes = (left: SkillFileTreeNode, right: SkillFileTreeNode) =>
  Number(isManifest(right)) - Number(isManifest(left)) ||
  Number(Boolean(right.file)) - Number(Boolean(left.file)) ||
  left.name.localeCompare(right.name);

export const buildSkillFileTree = (files: SkillFile[]): SkillFileTreeNode[] => {
  const root: SkillFileTreeNode = { name: '', path: '', children: [] };
  for (const file of files) {
    const segments = file.path.split('/').filter(Boolean);
    let parent = root;
    segments.forEach((segment, index) => {
      const isLeaf = index === segments.length - 1;
      const path = segments.slice(0, index + 1).join('/');
      let node = parent.children.find((child) => child.name === segment && Boolean(child.file) === isLeaf);
      if (!node) {
        node = { name: segment, path, children: [] };
        parent.children.push(node);
      }
      if (isLeaf) node.file = file;
      parent = node;
    });
  }
  const sortTree = (nodes: SkillFileTreeNode[]): SkillFileTreeNode[] =>
    nodes.map((node) => ({ ...node, children: sortTree(node.children) })).sort(compareNodes);
  return sortTree(root.children);
};

const PREVIEW_LANGUAGES = new Map<string, CodeSnippetLanguage>([
  ['py', 'python'],
  ['js', 'javascript'],
  ['mjs', 'javascript'],
  ['cjs', 'javascript'],
  ['jsx', 'javascript'],
  ['ts', 'javascript'],
  ['tsx', 'javascript'],
  ['json', 'json'],
  ['yaml', 'yaml'],
  ['yml', 'yaml'],
  ['sql', 'sql'],
  ['go', 'go'],
  ['java', 'java'],
]);

export const getPreviewLanguage = (path: string): CodeSnippetLanguage => {
  const extension = path.includes('.') ? path.slice(path.lastIndexOf('.') + 1).toLowerCase() : '';
  return PREVIEW_LANGUAGES.get(extension) ?? 'text';
};

export const formatFileSize = (bytes?: number) => {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

/** A size limit, without a trailing ".0" (25 MB rather than 25.0 MB). */
export const formatSizeLimit = (bytes: number) =>
  bytes < 1024 * 1024 ? formatFileSize(bytes) : `${Number((bytes / (1024 * 1024)).toFixed(1))} MB`;

// Text with NUL characters is almost certainly binary, which the preview cannot render.
export const looksBinary = (text: string) => text.includes('\u0000');
