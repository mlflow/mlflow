import { getCommonArtifacts } from './getCommonArtifacts';
import type { ArtifactListFilesResponse } from '../../../types';

describe('getCommonArtifacts', () => {
  it('returns an empty array if no runs are provided', () => {
    const result = getCommonArtifacts({});
    expect(result).toEqual([]);
  });

  it('returns the artifact list if only one run is provided', () => {
    const artifactsKeyedByRun = {
      run1: {
        files: [{ path: 'artifact1' }, { path: 'artifact2' }],
      } as ArtifactListFilesResponse,
    };

    const result = getCommonArtifacts(artifactsKeyedByRun);
    expect(result).toEqual(['artifact1', 'artifact2']);
  });

  it('returns common artifacts across multiple runs', () => {
    const artifactsKeyedByRun = {
      run1: {
        files: [{ path: 'artifact1' }, { path: 'artifact2' }],
      } as ArtifactListFilesResponse,
      run2: {
        files: [{ path: 'artifact1' }, { path: 'artifact3' }],
      } as ArtifactListFilesResponse,
    };

    const result = getCommonArtifacts(artifactsKeyedByRun);
    expect(result).toEqual(['artifact1']);
  });

  it('returns an empty array if no common artifacts exist in the given runs', () => {
    const artifactsKeyedByRun = {
      run1: {
        files: [{ path: 'artifact1' }, { path: 'artifact2' }],
      } as ArtifactListFilesResponse,
      run2: {
        files: [{ path: 'artifact3' }, { path: 'artifact4' }],
      } as ArtifactListFilesResponse,
    };

    const result = getCommonArtifacts(artifactsKeyedByRun);
    expect(result).toEqual([]);
  });

  it('works when there are some runs without any files', () => {
    const artifactsKeyedByRun = {
      run1: {
        files: [{ path: 'artifact1' }, { path: 'artifact2' }],
      } as ArtifactListFilesResponse,
      run2: {
        files: [] as any,
      } as ArtifactListFilesResponse,
    };

    const result = getCommonArtifacts(artifactsKeyedByRun);
    expect(result).toEqual([]);
  });

  it('filters out directories', () => {
    const artifactsKeyedByRun = {
      run1: {
        files: [{ path: 'artifact1', is_dir: true }],
      } as ArtifactListFilesResponse,
      run2: {
        files: [{ path: 'artifact1', is_dir: true }],
      } as ArtifactListFilesResponse,
    };

    const result = getCommonArtifacts(artifactsKeyedByRun);
    expect(result).toEqual([]);
  });
});
