import { extractArtifactPathFromModelSource } from './VersionUtils';

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
