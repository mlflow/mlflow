import { describe, expect, it } from '@jest/globals';

import {
  buildSkillFileTree,
  formatFileSize,
  formatSizeLimit,
  getPreviewLanguage,
  getSkillArtifactPath,
} from './skillFiles';

describe('skillFiles', () => {
  it('resolves the artifact path only for content MLflow stores', () => {
    expect(
      getSkillArtifactPath({ source_type: 'mlflow', source: 'mlflow-artifacts:/skills/demo/abc', subpath: null }),
    ).toBe('skills/demo/abc');
    expect(
      getSkillArtifactPath({
        source_type: 'mlflow',
        source: 'mlflow-artifacts:/plugins/pkg/abc/',
        subpath: 'skills/a',
      }),
    ).toBe('plugins/pkg/abc/skills/a');
    expect(
      getSkillArtifactPath({ source_type: 'git', source: 'https://github.com/acme/skills', subpath: 'a' }),
    ).toBeUndefined();
  });

  it('nests files under folders', () => {
    const tree = buildSkillFileTree([
      { path: 'docs/a.md', size: 1 },
      { path: 'SKILL.md', size: 2 },
    ]);
    expect(tree.map((node) => node.name)).toEqual(['SKILL.md', 'docs']);
    expect(tree[1].children.map((node) => node.path)).toEqual(['docs/a.md']);
  });

  it('maps extensions to highlighting and formats sizes', () => {
    expect(getPreviewLanguage('scripts/run.py')).toBe('python');
    expect(getPreviewLanguage('config.YML')).toBe('yaml');
    expect(getPreviewLanguage('SKILL.md')).toBe('text');
    expect(formatFileSize(900)).toBe('900 B');
    expect(formatFileSize(1536)).toBe('1.5 KB');
    expect(formatSizeLimit(25 * 1024 * 1024)).toBe('25 MB');
    expect(formatSizeLimit(1.5 * 1024 * 1024)).toBe('1.5 MB');
    expect(formatSizeLimit(64)).toBe('64 B');
  });
});
