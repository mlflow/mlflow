import { describe, it, expect } from '@jest/globals';
import { isVideoArtifactUrl } from './ImageGridPlot.common';

// getArtifactLocationUrl encodes the artifact path into the `path` query parameter,
// so the extension is never in the URL path. Reading location.pathname instead
// classifies every artifact as non-video.
const artifactUrl = (path: string) =>
  `/ajax-api/2.0/mlflow/get-artifact?path=${encodeURIComponent(path)}&run_uuid=${encodeURIComponent('abc123')}`;

const STEM = 'images/rollout+step+0+timestamp+1786553450112+c44e09fb-fa9c-416e-b4cd-75766917706e';

describe('isVideoArtifactUrl', () => {
  it.each(['mp4', 'webm', 'mov'])('detects a logged %s artifact', (extension) => {
    expect(isVideoArtifactUrl(artifactUrl(`${STEM}.${extension}`))).toBe(true);
  });

  it.each(['png', 'webp', 'gif', 'svg'])('does not treat a logged %s artifact as video', (extension) => {
    expect(isVideoArtifactUrl(artifactUrl(`${STEM}.${extension}`))).toBe(false);
  });

  it('is case insensitive', () => {
    expect(isVideoArtifactUrl(artifactUrl(`${STEM}.MP4`))).toBe(true);
  });

  it('handles the percent-encoded + delimiters in the stored filename', () => {
    const url = artifactUrl(`${STEM}.mp4`);
    expect(url).toContain('%2B');
    expect(isVideoArtifactUrl(url)).toBe(true);
  });

  it('falls back to the raw string when the URL cannot be parsed', () => {
    expect(isVideoArtifactUrl('rollout.mp4')).toBe(true);
    expect(isVideoArtifactUrl('rollout.png')).toBe(false);
  });

  it('does not match when the extension is absent', () => {
    expect(isVideoArtifactUrl(artifactUrl('images/rollout'))).toBe(false);
  });
});
