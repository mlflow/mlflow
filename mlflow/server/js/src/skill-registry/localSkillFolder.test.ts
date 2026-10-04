import { describe, expect, it } from '@jest/globals';
import { gunzipSync } from 'zlib';
import { createSkillTar, findSkillManifest, gzip, readSkillManifest } from './localSkillFolder';

const folderFile = (path: string) => {
  const file = new File(['# Skill\n'], path.split('/').pop() ?? path);
  Object.defineProperty(file, 'webkitRelativePath', { value: path });
  return file;
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
});
