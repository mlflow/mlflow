import { describe, it, expect } from '@jest/globals';
import { extractArtifactPathFromModelSource, extractLoggedModelIdFromModelSource } from './VersionUtils';

describe('extractArtifactPathFromModelSource', () => {
  it('test extractArtifactPathFromModelSource', () => {
    expect(extractArtifactPathFromModelSource('mlflow-artifacts:/0/01bcd/artifacts/xx/yy', '01bcd')).toBe('xx/yy');
    expect(extractArtifactPathFromModelSource('mlflow-artifacts:/0/01bcd/artifacts/artifacts/xx/yy', '01bcd')).toBe(
      'artifacts/xx/yy',
    );
    expect(extractArtifactPathFromModelSource('mlflow-artifacts:/0/01bcd/artifacts/xx/yy', '01bce')).toBe(undefined);
    expect(extractArtifactPathFromModelSource('file///path/to/mlruns/0/01bcd/artifacts/xx/yy', '01bcd')).toBe('xx/yy');
    expect(extractArtifactPathFromModelSource('file///path/to/artifacts/mlruns/0/01bcd/artifacts/xx/yy', '01bcd')).toBe(
      'xx/yy',
    );
    expect(extractArtifactPathFromModelSource('file///path/to/mlruns/0/01bcd/artifacts/artifacts/xx/yy', '01bcd')).toBe(
      'artifacts/xx/yy',
    );
    expect(extractArtifactPathFromModelSource('file///path/to/mlruns/0/01bcd/artifacts/xx/yy', '01bce')).toBe(
      undefined,
    );
  });
});

describe('extractLoggedModelIdFromModelSource', () => {
  it('returns the model ID only for `models:/<model_id>` sources', () => {
    expect(extractLoggedModelIdFromModelSource('models:/m-1bc96c322a8441c7a5af8de3df2cc4ce')).toBe(
      'm-1bc96c322a8441c7a5af8de3df2cc4ce',
    );
    expect(extractLoggedModelIdFromModelSource('models:/Model B/2')).toBe(undefined);
    expect(extractLoggedModelIdFromModelSource('models:/Model B@champion')).toBe(undefined);
    expect(extractLoggedModelIdFromModelSource('mlflow-artifacts:/0/01bcd/artifacts/model')).toBe(undefined);
    expect(extractLoggedModelIdFromModelSource(undefined)).toBe(undefined);
  });
});
