const BLOCK = 512;
const encoder = new TextEncoder();

export interface SkillManifestFields {
  name?: string;
  description?: string;
}

const frontmatterValue = (frontmatter: string, key: string) => {
  const match = frontmatter.match(new RegExp(`^${key}:\\s*(.+)\\s*$`, 'm'));
  return match?.[1]?.trim().replace(/^['"]|['"]$/g, '');
};

export const readSkillManifest = (content: string): SkillManifestFields => {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match?.[1]) return {};
  const name = frontmatterValue(match[1], 'name');
  const description = frontmatterValue(match[1], 'description');
  return {
    ...(name ? { name } : {}),
    ...(description ? { description } : {}),
  };
};

const relativePath = (file: File) => file.webkitRelativePath || file.name;

export const findSkillManifest = (files: File[]) =>
  files
    .filter((file) => relativePath(file).split('/').pop() === 'SKILL.md')
    .sort((left, right) => relativePath(left).split('/').length - relativePath(right).split('/').length)[0];

const archivePath = (file: File, files: File[]) => {
  const paths = files.map(relativePath).filter(Boolean);
  const roots = new Set(paths.map((path) => path.split('/')[0]));
  const stripRoot = roots.size === 1 && paths.every((path) => path.includes('/'));
  const path = relativePath(file);
  const stripped = stripRoot ? path.split('/').slice(1).join('/') : path;
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

export const createSkillTar = (entries: { name: string; bytes: Uint8Array }[]) => {
  const parts = entries.flatMap(({ name, bytes }) => {
    const padding = (BLOCK - (bytes.length % BLOCK)) % BLOCK;
    return [tarHeader(name, bytes.length), bytes, new Uint8Array(padding)];
  });
  const end = new Uint8Array(BLOCK * 2);
  const size = parts.reduce((total, part) => total + part.length, 0) + end.length;
  const archive = new Uint8Array(size);
  let offset = 0;
  [...parts, end].forEach((part) => {
    archive.set(part, offset);
    offset += part.length;
  });
  return archive;
};

const crc32 = (bytes: Uint8Array) => {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
};

export const gzipStored = (bytes: Uint8Array) => {
  const chunks: Uint8Array[] = [new Uint8Array([0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff])];
  const maxBlock = 0xffff;
  for (let offset = 0; offset < bytes.length || offset === 0; offset += maxBlock) {
    const end = Math.min(offset + maxBlock, bytes.length);
    const block = bytes.subarray(offset, end);
    const last = end === bytes.length;
    const header = new Uint8Array(5);
    header[0] = last ? 0x01 : 0x00;
    header[1] = block.length & 0xff;
    header[2] = (block.length >> 8) & 0xff;
    const complement = ~block.length & 0xffff;
    header[3] = complement & 0xff;
    header[4] = (complement >> 8) & 0xff;
    chunks.push(header, block);
    if (bytes.length === 0) break;
  }
  const checksum = crc32(bytes);
  const trailer = new Uint8Array(8);
  trailer[0] = checksum & 0xff;
  trailer[1] = (checksum >> 8) & 0xff;
  trailer[2] = (checksum >> 16) & 0xff;
  trailer[3] = (checksum >> 24) & 0xff;
  trailer[4] = bytes.length & 0xff;
  trailer[5] = (bytes.length >> 8) & 0xff;
  trailer[6] = (bytes.length >> 16) & 0xff;
  trailer[7] = (bytes.length >> 24) & 0xff;
  chunks.push(trailer);
  const size = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const gzip = new Uint8Array(size);
  let write = 0;
  chunks.forEach((chunk) => {
    gzip.set(chunk, write);
    write += chunk.length;
  });
  return gzip;
};

export const packageSkillFolder = async (files: File[]) => {
  const entries = await Promise.all(
    files.map(async (file) => ({
      name: archivePath(file, files),
      bytes: new Uint8Array(await file.arrayBuffer()),
    })),
  );
  const tar = createSkillTar(entries.filter((entry) => entry.name));
  return new Blob([gzipStored(tar)], { type: 'application/gzip' });
};
