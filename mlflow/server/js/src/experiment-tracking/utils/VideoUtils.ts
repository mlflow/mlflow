import { getBasename } from '../../common/utils/FileUtils';

/**
 * Video artifact extensions used for the video gallery panel.
 * This is broader than VIDEO_EXTENSIONS in FileUtils (which drives single-artifact preview)
 * because the gallery should detect .ogg/.ogv which can contain video content.
 */
export const VIDEO_ARTIFACT_EXTENSIONS = new Set(['mp4', 'webm', 'mov', 'm4v', 'ogg', 'ogv', 'mkv', 'avi']);

/**
 * Directories known to commonly contain video artifacts (e.g. from RL rollouts).
 * The hook will proactively scan these directories for video content.
 */
export const VIDEO_ARTIFACT_DIRECTORIES = ['videos', 'media/videos', 'media', 'rollouts'];

/**
 * Check whether an artifact path points to a video file.
 * Case-insensitive extension matching.
 */
export function isVideoArtifactPath(path: string): boolean {
  const parts = path.split(/[./]/);
  const ext = parts[parts.length - 1]?.toLowerCase();
  return VIDEO_ARTIFACT_EXTENSIONS.has(ext);
}

/**
 * Extract a numeric step/episode number from a filename if present.
 * Matches patterns like: step_100.mp4, rollout_042.webm, episode-5.mp4, 0003.mp4
 * Returns null if no number can be inferred.
 */
export function extractStepNumber(filename: string): number | null {
  // Strip extension
  const base = filename.replace(/\.[^.]+$/, '');
  // Try common patterns: trailing number after separator
  const match = base.match(/[-_](\d+)$/);
  if (match) {
    return parseInt(match[1], 10);
  }
  // Try: filename is purely numeric
  if (/^\d+$/.test(base)) {
    return parseInt(base, 10);
  }
  return null;
}

/**
 * Sort video artifact paths in a useful order:
 * - If step numbers can be inferred, sort numerically by step.
 * - Otherwise sort alphabetically by path.
 */
export function sortVideoArtifacts(paths: string[]): string[] {
  return [...paths].sort((a, b) => {
    const stepA = extractStepNumber(getBasename(a));
    const stepB = extractStepNumber(getBasename(b));
    if (stepA !== null && stepB !== null) {
      return stepA - stepB;
    }
    // If only one has a step number, it comes first
    if (stepA !== null) return -1;
    if (stepB !== null) return 1;
    // Fallback: alphabetical by full path
    return a.localeCompare(b);
  });
}
