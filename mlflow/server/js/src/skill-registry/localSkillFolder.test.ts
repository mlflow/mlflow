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
});
