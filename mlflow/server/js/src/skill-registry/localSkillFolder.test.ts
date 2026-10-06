import { describe, expect, it } from '@jest/globals';
import { gunzipSync } from 'zlib';
import { createSkillTar, findSkillManifest, gzip, packageSkillFolder, readSkillManifest } from './localSkillFolder';

const folderFile = (path: string, content = '# Skill\n') => {
  const file = new File([content], path.split('/').pop() ?? path);
  const bytes = new TextEncoder().encode(content);
  // jsdom's File has no arrayBuffer().
  Object.defineProperties(file, {
    webkitRelativePath: { value: path },
    arrayBuffer: { value: async () => bytes.buffer },
  });
  return file;
};

const readBlob = (blob: Blob) =>
  new Promise<Buffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(Buffer.from(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });

// Entry names and contents of an uncompressed ustar archive.
const listTar = (tar: Buffer) => {
  const entries: Record<string, string> = {};
  for (let offset = 0; offset + 512 <= tar.length; ) {
    const field = (start: number, length: number) =>
      tar
        .subarray(offset + start, offset + start + length)
        .toString('utf8')
        .replace(/\0.*$/s, '');
    const name = field(0, 100);
    if (!name) break;
    const prefix = field(345, 155);
    const size = parseInt(field(124, 12).trim(), 8);
    entries[prefix ? `${prefix}/${name}` : name] = tar.subarray(offset + 512, offset + 512 + size).toString('utf8');
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  return entries;
};

describe('localSkillFolder', () => {
  it('reads the skill name and description from SKILL.md frontmatter', () => {
    expect(readSkillManifest('---\nname: code-review\ndescription: Reviews pull requests\n---\n# Skill\n')).toEqual({
      name: 'code-review',
      description: 'Reviews pull requests',
    });
  });

  it('reads block-scalar and quoted frontmatter values', () => {
    expect(readSkillManifest('---\nname: "code-review"\ndescription: >\n  Reviews pull\n  requests\n---\n')).toEqual({
      name: 'code-review',
      description: 'Reviews pull requests',
    });
    expect(readSkillManifest('---\nname: [unclosed\n---\n')).toEqual({});
  });

  it('finds SKILL.md at the root of the selected folder', () => {
    const manifest = folderFile('code-review/SKILL.md');
    expect(findSkillManifest([folderFile('code-review/scripts/run.sh'), manifest])).toBe(manifest);
  });

  it('ignores a SKILL.md nested below the archive root', () => {
    expect(findSkillManifest([folderFile('project/README.md'), folderFile('project/skills/a/SKILL.md')])).toBe(
      undefined,
    );
  });

  it('packs SKILL.md at the archive root', async () => {
    const archive = createSkillTar([{ name: 'SKILL.md', bytes: new TextEncoder().encode('# Skill\n') }]);
    const unpacked = gunzipSync(await gzip(archive));
    expect(unpacked.toString('utf8')).toContain('SKILL.md');
  });

  it('stores a path longer than 100 bytes in the ustar prefix', async () => {
    const path = `${'docs'.repeat(30)}/SKILL.md`;
    const archive = createSkillTar([{ name: path, bytes: new TextEncoder().encode('# Skill\n') }]);
    const unpacked = gunzipSync(await gzip(archive));
    const field = (start: number, length: number) =>
      new TextDecoder().decode(unpacked.subarray(start, start + length)).replace(/\0+$/, '');
    expect(`${field(345, 155)}/${field(0, 100)}`).toBe(path);
  });

  it('compresses tar framing for many small files', async () => {
    const entries = Array.from({ length: 200 }, (_, index) => ({
      name: `references/file-${index}.md`,
      bytes: new TextEncoder().encode(`# Reference ${index}\n`),
    }));
    const archive = createSkillTar(entries);
    const compressed = await gzip(archive);
    expect(compressed.length).toBeLessThan(archive.length / 10);
    expect(Buffer.from(gunzipSync(compressed)).equals(Buffer.from(archive))).toBe(true);
  });

  it('rejects a path that cannot fit in a ustar header', () => {
    expect(() => createSkillTar([{ name: 'a'.repeat(101), bytes: new Uint8Array() }])).toThrow(/too long/);
  });

  it('packages a picked folder without its own name, keeping nested paths', async () => {
    const content = await packageSkillFolder([
      folderFile('my-skill/SKILL.md', '---\nname: my-skill\n---\n'),
      folderFile('my-skill/scripts/run.py', 'print(1)\n'),
    ]);

    expect(content.type).toBe('application/gzip');
    expect(listTar(gunzipSync(await readBlob(content)))).toEqual({
      'SKILL.md': '---\nname: my-skill\n---\n',
      'scripts/run.py': 'print(1)\n',
    });
  });

  it('refuses to package a path that climbs out of the folder', async () => {
    await expect(
      packageSkillFolder([folderFile('my-skill/SKILL.md'), folderFile('my-skill/../secret.txt')]),
    ).rejects.toThrow("Cannot package 'my-skill/../secret.txt'.");
  });
});
