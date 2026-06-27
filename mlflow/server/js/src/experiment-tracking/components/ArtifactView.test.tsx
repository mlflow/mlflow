import { ArtifactViewImpl } from './ArtifactView';

describe('ArtifactViewImpl#getExistingModelVersions', () => {
  const makeInstance = (props: any, state: any) => {
    const instance = Object.create(ArtifactViewImpl.prototype);
    instance.props = props;
    instance.state = state;
    instance.getActiveNodeRealPath = () =>
      state.activeNodeId
        ? `${props.artifactRootUri}/${state.activeNodeId}`
        : props.artifactRootUri;
    return instance;
  };

  it('returns versions keyed by raw S3 path (legacy)', () => {
    const version = [{ source: 's3://bucket/run123/artifacts/model', version: '1' }];
    const instance = makeInstance(
      {
        artifactRootUri: 's3://bucket/run123/artifacts',
        modelVersionsBySource: { 's3://bucket/run123/artifacts/model': version },
        runUuid: 'run123',
      },
      { activeNodeId: 'model' },
    );
    expect(instance.getExistingModelVersions()).toBe(version);
  });

  it('returns versions keyed by runs:/ URI (UI registration after #18501)', () => {
    const version = [{ source: 'runs:/run123/model', version: '1' }];
    const instance = makeInstance(
      {
        artifactRootUri: 's3://bucket/run123/artifacts',
        modelVersionsBySource: { 'runs:/run123/model': version },
        runUuid: 'run123',
      },
      { activeNodeId: 'model' },
    );
    expect(instance.getExistingModelVersions()).toBe(version);
  });

  it('returns undefined when no matching version exists', () => {
    const instance = makeInstance(
      {
        artifactRootUri: 's3://bucket/run123/artifacts',
        modelVersionsBySource: {},
        runUuid: 'run123',
      },
      { activeNodeId: 'model' },
    );
    expect(instance.getExistingModelVersions()).toBeUndefined();
  });
});