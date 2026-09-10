import { useState, useEffect, useCallback } from 'react';
import { MlflowService } from '../../../sdk/MlflowService';
import { isVideoArtifactPath, sortVideoArtifacts, VIDEO_ARTIFACT_DIRECTORIES } from '../../../utils/VideoUtils';

interface ArtifactFileInfo {
  path: string;
  is_dir: boolean;
  file_size?: number;
}

/**
 * Hook that discovers video artifacts for a given run.
 * Scans the root artifact directory and known video directories
 * (videos/, media/, media/videos/, rollouts/) for video files.
 */
export function useRunVideoArtifacts(runUuid: string): {
  videos: string[];
  loading: boolean;
} {
  const [videos, setVideos] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchArtifacts = useCallback(async () => {
    try {
      const videoPaths: string[] = [];
      const scannedDirs = new Set<string>();

      const listDir = async (path?: string): Promise<ArtifactFileInfo[]> => {
        const dirKey = path || '';
        if (scannedDirs.has(dirKey)) return [];
        scannedDirs.add(dirKey);

        try {
          const response = await MlflowService.listArtifacts({
            run_uuid: runUuid,
            ...(path && { path }),
          });
          return response.files || [];
        } catch {
          // Directory may not exist, that's OK
          return [];
        }
      };

      // Scan root directory
      const rootFiles = await listDir();
      for (const file of rootFiles) {
        if (!file.is_dir && isVideoArtifactPath(file.path)) {
          videoPaths.push(file.path);
        }
      }

      // Scan known video directories (and any root-level dirs that match)
      const dirsToScan = [...VIDEO_ARTIFACT_DIRECTORIES];
      for (const file of rootFiles) {
        if (file.is_dir) {
          const dirName = file.path.split('/').pop() || file.path;
          if (
            VIDEO_ARTIFACT_DIRECTORIES.some(
              (vd) => vd === dirName || vd.startsWith(dirName + '/'),
            )
          ) {
            dirsToScan.push(file.path);
          }
        }
      }

      // Fetch contents of known directories in parallel
      const dirResults = await Promise.all(
        [...new Set(dirsToScan)].map((dir) => listDir(dir)),
      );

      for (const files of dirResults) {
        for (const file of files) {
          if (!file.is_dir && isVideoArtifactPath(file.path)) {
            videoPaths.push(file.path);
          } else if (file.is_dir) {
            // One level deeper for nested directories like media/videos/
            const nestedFiles = await listDir(file.path);
            for (const nested of nestedFiles) {
              if (!nested.is_dir && isVideoArtifactPath(nested.path)) {
                videoPaths.push(nested.path);
              }
            }
          }
        }
      }

      // Deduplicate and sort
      const uniquePaths = [...new Set(videoPaths)];
      setVideos(sortVideoArtifacts(uniquePaths));
    } catch {
      // If artifact listing fails entirely, show no videos
      setVideos([]);
    } finally {
      setLoading(false);
    }
  }, [runUuid]);

  useEffect(() => {
    fetchArtifacts();
  }, [fetchArtifacts]);

  return { videos, loading };
}
