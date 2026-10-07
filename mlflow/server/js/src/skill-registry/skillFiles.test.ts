import { describe, expect, it, jest } from '@jest/globals';

import { SkillRegistryApi } from './api';

import {
  buildSkillFileTree,
  formatFileSize,
  formatSizeLimit,
  getPreviewLanguage,
  getSkillArtifactPath,
  listSkillFiles,
  MAX_LISTED_SKILL_DIRECTORIES,
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

  it('stops listing after the directory cap even when no files were found', async () => {
    // A root with 150 empty directories.
    const listArtifacts = jest
      .spyOn(SkillRegistryApi, 'listArtifacts')
      .mockImplementation(async (path: string) =>
        path === 'root'
          ? { files: Array.from({ length: 150 }, (_, index) => ({ path: `dir-${index}`, is_dir: true })) }
          : {},
      );

    const result = await listSkillFiles('root');

    expect(listArtifacts).toHaveBeenCalledTimes(MAX_LISTED_SKILL_DIRECTORIES);
    expect(result).toEqual({ files: [], truncated: true });
    listArtifacts.mockRestore();
  });

  it('lists an empty file, which the listing gives no size, as 0 bytes', async () => {
    const listArtifacts = jest
      .spyOn(SkillRegistryApi, 'listArtifacts')
      .mockResolvedValue({ files: [{ path: 'empty.txt', is_dir: false }] });

    const result = await listSkillFiles('root');

    expect(result.files).toEqual([{ path: 'empty.txt', size: 0 }]);
    expect(formatFileSize(result.files[0].size)).toBe('0 B');
    listArtifacts.mockRestore();
  });

  it('lists the directories of a level in parallel', async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const listArtifacts = jest.spyOn(SkillRegistryApi, 'listArtifacts').mockImplementation(async (path: string) => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 5));
      inFlight -= 1;
      if (path === 'root') return { files: ['a', 'b', 'c'].map((name) => ({ path: name, is_dir: true })) };
      return { files: [{ path: 'SKILL.md', is_dir: false, file_size: 1 }] };
    });

    const result = await listSkillFiles('root');

    expect(maxInFlight).toBe(3);
    expect(result.files.map((file) => file.path).sort()).toEqual(['a/SKILL.md', 'b/SKILL.md', 'c/SKILL.md']);
    expect(result.truncated).toBe(false);
    listArtifacts.mockRestore();
  });
});
