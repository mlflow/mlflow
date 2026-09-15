import { describe, expect, it } from '@jest/globals';

import { getTraceArtifactLocation, getTraceAttachmentUrl } from './attachment-utils';

const PROXY_ROOT = `${window.location.origin}/api/2.0/mlflow-artifacts/artifacts/1/traces/tr-abc/artifacts`;
const TRACKING_URL = 'ajax-api/2.0/mlflow/get-trace-artifact?request_id=tr-abc&path=att-1';

describe('getTraceAttachmentUrl', () => {
  it('falls back to the tracking endpoint when no artifact location is known', () => {
    expect(getTraceAttachmentUrl('tr-abc', 'att-1')).toContain(TRACKING_URL);
  });

  it('falls back to the tracking endpoint for a non-proxy artifact location', () => {
    expect(getTraceAttachmentUrl('tr-abc', 'att-1', 'mlflow-artifacts:/1/traces/tr-abc/artifacts')).toContain(
      TRACKING_URL,
    );
  });

  it('falls back to the tracking endpoint for a cross-origin artifact location', () => {
    expect(
      getTraceAttachmentUrl('tr-abc', 'att-1', 'https://evil.example.com/api/2.0/mlflow-artifacts/artifacts/1'),
    ).toContain(TRACKING_URL);
  });

  it('uses the stored proxy URI when it is eligible', () => {
    expect(getTraceAttachmentUrl('tr-abc', 'att-1', PROXY_ROOT)).toBe(`${PROXY_ROOT}/attachments/att-1`);
  });

  it('encodes attachment ids that contain reserved characters', () => {
    expect(getTraceAttachmentUrl('tr-abc', 'a b#c', PROXY_ROOT)).toBe(`${PROXY_ROOT}/attachments/a%20b%23c`);
  });
});

describe('getTraceArtifactLocation', () => {
  it('reads the artifact location from array-shaped tags', () => {
    expect(getTraceArtifactLocation({ tags: [{ key: 'mlflow.artifactLocation', value: PROXY_ROOT }] })).toBe(
      PROXY_ROOT,
    );
  });

  it('reads the artifact location from object-shaped tags', () => {
    expect(getTraceArtifactLocation({ tags: { 'mlflow.artifactLocation': PROXY_ROOT } })).toBe(PROXY_ROOT);
  });

  it('returns undefined when the tag is absent', () => {
    expect(getTraceArtifactLocation({ tags: [] })).toBeUndefined();
    expect(getTraceArtifactLocation({})).toBeUndefined();
    expect(getTraceArtifactLocation(undefined)).toBeUndefined();
  });
});
