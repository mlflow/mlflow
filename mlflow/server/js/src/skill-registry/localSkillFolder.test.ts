import { describe, expect, it } from '@jest/globals';
import { gunzipSync } from 'zlib';
import { createSkillTar, gzipStored, readSkillManifest } from './localSkillFolder';

describe('localSkillFolder', () => {
  it('reads the skill name and description from SKILL.md frontmatter', () => {
    expect(readSkillManifest('---\nname: code-review\ndescription: Reviews pull requests\n---\n# Skill\n')).toEqual({
      name: 'code-review',
      description: 'Reviews pull requests',
    });
  });

  it('packs SKILL.md at the archive root', () => {
    const archive = createSkillTar([{ name: 'SKILL.md', bytes: new TextEncoder().encode('# Skill\n') }]);
    const unpacked = gunzipSync(gzipStored(archive));
    expect(unpacked.toString('utf8')).toContain('SKILL.md');
  });

  it('stores a path longer than 100 bytes in the ustar prefix', () => {
    const path = `${'docs'.repeat(30)}/SKILL.md`;
    const archive = createSkillTar([{ name: path, bytes: new TextEncoder().encode('# Skill\n') }]);
    const unpacked = gunzipSync(gzipStored(archive));
    const field = (start: number, length: number) =>
      new TextDecoder().decode(unpacked.subarray(start, start + length)).replace(/\0+$/, '');
    expect(`${field(345, 155)}/${field(0, 100)}`).toBe(path);
  });

  it('rejects a path that cannot fit in a ustar header', () => {
    expect(() => createSkillTar([{ name: 'a'.repeat(101), bytes: new Uint8Array() }])).toThrow(/too long/);
  });
});
