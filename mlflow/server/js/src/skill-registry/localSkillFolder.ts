import yaml from 'js-yaml';

const BLOCK = 512;
const encoder = new TextEncoder();

export interface SkillManifestFields {
  name?: string;
  description?: string;
}

const stringField = (frontmatter: unknown, key: string) => {
  if (!frontmatter || typeof frontmatter !== 'object') return undefined;
  const value = (frontmatter as Record<string, unknown>)[key];
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
};

export const readSkillManifest = (content: string): SkillManifestFields => {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match?.[1]) return {};
  let frontmatter: unknown;
  try {
    frontmatter = yaml.safeLoad(match[1]);
  } catch {
    return {};
  }
  const name = stringField(frontmatter, 'name');
  const description = stringField(frontmatter, 'description');
  return {
    ...(name ? { name } : {}),
    ...(description ? { description } : {}),
  };
};

const relativePath = (file: File) => file.webkitRelativePath || file.name;

export const totalFileSize = (files: File[]) => files.reduce((total, file) => total + file.size, 0);

/** The server limit a selected folder exceeds, judged from file sizes and count alone so nothing is read. */
export const exceededContentLimit = (
  files: File[],
  { maxBytes, maxFiles }: { maxBytes?: number; maxFiles?: number },
) => {
  if (maxFiles !== undefined && files.length > maxFiles) return 'files' as const;
  if (maxBytes !== undefined && totalFileSize(files) > maxBytes) return 'bytes' as const;
  return undefined;
};

// A directory picker prefixes every path with the selected folder's name, which is not part of the skill.
const sharesSingleRoot = (files: File[]) => {
  const paths = files.map(relativePath).filter(Boolean);
  const roots = new Set(paths.map((path) => path.split('/')[0]));
  return roots.size === 1 && paths.every((path) => path.includes('/'));
};

const strippedPath = (file: File, stripRoot: boolean) => {
  const path = relativePath(file);
  return stripRoot ? path.split('/').slice(1).join('/') : path;
};

// The upload has no subpath, so only a SKILL.md at the archive root makes a usable skill.
export const findSkillManifest = (files: File[]) => {
  const stripRoot = sharesSingleRoot(files);
  return files.find((file) => strippedPath(file, stripRoot) === 'SKILL.md');
};

const archivePath = (file: File, stripRoot: boolean) => {
  const path = relativePath(file);
  const stripped = strippedPath(file, stripRoot);
  if (!stripped || stripped.split('/').some((segment) => segment === '..' || segment === '.')) {
    throw new Error(`Cannot package '${path}'.`);
  }
  return stripped;
};

const octal = (value: number, length: number) => {
  const text = value.toString(8).padStart(length - 1, '0');
  const bytes = new Uint8Array(length);
  for (let index = 0; index < length - 1; index += 1) bytes[index] = text.charCodeAt(index);
  return bytes;
};

const splitUstarName = (path: string) => {
  const bytes = encoder.encode(path);
  if (bytes.length <= 100) {
    return { name: bytes, prefix: new Uint8Array() };
  }
  for (let index = bytes.length - 1; index >= 0; index -= 1) {
    if (bytes[index] !== 0x2f) continue;
    const prefixLength = index;
    const nameLength = bytes.length - index - 1;
    if (prefixLength > 0 && prefixLength <= 155 && nameLength > 0 && nameLength <= 100) {
      return { name: bytes.subarray(index + 1), prefix: bytes.subarray(0, index) };
    }
  }
  return undefined;
};

const tarHeader = (name: string, size: number) => {
  const split = splitUstarName(name);
  if (!split) {
    throw new Error(`Cannot package '${name}'. The path is too long for a tar archive.`);
  }
  const header = new Uint8Array(BLOCK);
  header.set(split.name);
  header.set(octal(0o644, 8), 100);
  header.set(octal(0, 8), 108);
  header.set(octal(0, 8), 116);
  header.set(octal(size, 12), 124);
  header.set(octal(0, 12), 136);
  header.set(encoder.encode('        '), 148);
  header[156] = '0'.charCodeAt(0);
  header.set(encoder.encode('ustar'), 257);
  header.set(encoder.encode('00'), 263);
  header.set(split.prefix, 345);
  const sum = header.reduce((total, byte) => total + byte, 0);
  const checksum = sum.toString(8).padStart(6, '0');
  for (let index = 0; index < 6; index += 1) header[148 + index] = checksum.charCodeAt(index);
  header[154] = 0;
  header[155] = ' '.charCodeAt(0);
  return header;
};

const concatBytes = (parts: Uint8Array[]) => {
  const result = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
};

export const createSkillTar = (entries: { name: string; bytes: Uint8Array }[]) =>
  concatBytes([
    ...entries.flatMap(({ name, bytes }) => {
      const padding = (BLOCK - (bytes.length % BLOCK)) % BLOCK;
      return [tarHeader(name, bytes.length), bytes, new Uint8Array(padding)];
    }),
    new Uint8Array(BLOCK * 2),
  ]);

// Imported lazily to keep pako out of the main bundle, as StringUtils does.
const lazyPako = () => import('pako');

// Real compression matters: the server caps the upload at the decompressed limit plus a small slack, and
// tar headers and block padding for many small files would otherwise eat that slack.
// pako allocates a fresh ArrayBuffer; its typings only promise ArrayBufferLike, which Blob rejects.
export const gzip = async (bytes: Uint8Array) => (await lazyPako()).gzip(bytes) as Uint8Array<ArrayBuffer>;

export const packageSkillFolder = async (files: File[]) => {
  const stripRoot = sharesSingleRoot(files);
  const entries = await Promise.all(
    files.map(async (file) => ({
      name: archivePath(file, stripRoot),
      bytes: new Uint8Array(await file.arrayBuffer()),
    })),
  );
  return new Blob([await gzip(createSkillTar(entries))], { type: 'application/gzip' });
};
