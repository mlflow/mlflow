import {
  artifactUriToLocalPath,
  getArtifactUriScheme,
  isLocalArtifactUri,
} from '../../../src/core/utils/artifact_uri';

describe('artifact_uri helpers', () => {
  describe('getArtifactUriScheme', () => {
    it('returns the scheme for URIs that have one', () => {
      expect(getArtifactUriScheme('mlflow-artifacts:/0/traces/tr-1/artifacts')).toBe(
        'mlflow-artifacts',
      );
      expect(getArtifactUriScheme('file:///tmp/a/b')).toBe('file');
      expect(getArtifactUriScheme('s3://bucket/key')).toBe('s3');
      expect(getArtifactUriScheme('S3://bucket/key')).toBe('s3');
    });

    it('returns an empty string for bare filesystem paths', () => {
      expect(getArtifactUriScheme('/tmp/mlflow-artifacts/0/traces/tr-1/artifacts')).toBe('');
      expect(getArtifactUriScheme('relative/path')).toBe('');
    });

    it('treats a single-letter (Windows drive) scheme as a bare path', () => {
      expect(getArtifactUriScheme('C:\\tmp\\artifacts')).toBe('');
      expect(getArtifactUriScheme('d:/tmp/artifacts')).toBe('');
    });
  });

  describe('isLocalArtifactUri', () => {
    it('treats bare paths and file:// URIs as local', () => {
      expect(isLocalArtifactUri('/tmp/mlflow-artifacts/0')).toBe(true);
      expect(isLocalArtifactUri('file:///tmp/mlflow-artifacts/0')).toBe(true);
      expect(isLocalArtifactUri('C:\\tmp\\artifacts')).toBe(true);
    });

    it('treats mlflow-artifacts and remote schemes as non-local', () => {
      expect(isLocalArtifactUri('mlflow-artifacts:/0/traces/tr-1/artifacts')).toBe(false);
      expect(isLocalArtifactUri('s3://bucket/key')).toBe(false);
      expect(isLocalArtifactUri('gs://bucket/key')).toBe(false);
    });
  });

  describe('artifactUriToLocalPath', () => {
    it('returns bare paths unchanged', () => {
      expect(artifactUriToLocalPath('/tmp/mlflow-artifacts/0/traces/tr-1/artifacts')).toBe(
        '/tmp/mlflow-artifacts/0/traces/tr-1/artifacts',
      );
    });

    it('decodes file:// URIs to filesystem paths', () => {
      expect(artifactUriToLocalPath('file:///tmp/mlflow-artifacts/0')).toBe(
        '/tmp/mlflow-artifacts/0',
      );
      // Percent-encoded spaces are decoded.
      expect(artifactUriToLocalPath('file:///tmp/with%20space/0')).toBe('/tmp/with space/0');
    });
  });
});
