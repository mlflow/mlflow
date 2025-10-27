import { listArtifactsApi } from '../../../actions';
import { useRunsArtifacts } from './useRunsArtifacts';
import type { ArtifactListFilesResponse } from '../../../types';
import { renderHook, cleanup, waitFor } from '@testing-library/react';

const mockArtifactsData: Record<string, ArtifactListFilesResponse> = {
  'run-1': {
    root_uri: 'run-1',
    files: [
      {
        path: 'artifact1.txt',
        is_dir: false,
        file_size: 300,
      },
    ],
  },
  'run-2': {
    root_uri: 'run-2',
    files: [
      {
        path: 'artifact2.txt',
        is_dir: false,
        file_size: 300,
      },
    ],
  },
};

jest.mock('../../../actions', () => ({
  ...jest.requireActual<typeof import('../../../actions')>('../../../actions'),
  listArtifactsApi: jest.fn((runUuid: string) => {
    return {
      payload: mockArtifactsData[runUuid],
    };
  }),
}));

describe('useRunsArtifacts', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    cleanup();
  });

  test('fetches artifacts for given run UUIDs', async () => {
    const runUuids = ['run-1', 'run-2'];
    const { result } = renderHook(() => useRunsArtifacts(runUuids));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Make sure API is called correctly
    expect(listArtifactsApi).toHaveBeenCalledWith('run-1');
    expect(listArtifactsApi).toHaveBeenCalledWith('run-2');
    expect(listArtifactsApi).toHaveBeenCalledTimes(2);

    expect(result.current.artifactsKeyedByRun).toEqual(mockArtifactsData);
  });

  test('returns empty object when no run UUIDs are provided', () => {
    const runUuids: string[] = [];
    const { result } = renderHook(() => useRunsArtifacts(runUuids));

    expect(result.current.artifactsKeyedByRun).toEqual({});
  });

  test('returns empty object when no artifacts are found', () => {
    const runUuids = ['run-3'];
    const { result } = renderHook(() => useRunsArtifacts(runUuids));

    expect(result.current.artifactsKeyedByRun).toEqual({});
  });
});
