import { randomUUID } from 'crypto';
import {
  createAuthProvider,
  Issue,
  IssueCategory,
  IssueSeverity,
  IssueStatus,
  MlflowClient,
} from '../../src';
import { makeRequest } from '../../src/clients/utils';
import { TEST_TRACKING_URI } from '../helper';

describe('MlflowClient issues against OSS', () => {
  const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
  const client = new MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });
  let experimentId: string;

  beforeEach(async () => {
    experimentId = await client.createExperiment(`typescript-issues-${randomUUID()}`);
  });

  afterEach(async () => {
    if (experimentId) {
      await client.deleteExperiment(experimentId);
    }
  });

  it('creates and gets a caller-assigned issue and rejects duplicate IDs', async () => {
    const options = {
      experimentId,
      issueId: `iss-${randomUUID().replace(/-/g, '')}`,
      name: 'Incorrect answer',
      description: 'The response contradicts the source.',
      severity: IssueSeverity.HIGH,
      categories: [IssueCategory.CORRECTNESS, 'custom-category'],
      lastSeenTimestamp: Date.now() - 5000,
      metadata: { owner: 'quality' },
    };
    const created = await client.createIssue(options);
    expect(created).toBeInstanceOf(Issue);
    expect(created).toMatchObject({
      ...options,
      status: IssueStatus.PENDING,
      createdTimestamp: expect.any(Number),
      lastUpdatedTimestamp: expect.any(Number),
    });

    const retrieved = await client.getIssue(created.issueId, experimentId);
    expect(retrieved.toJson()).toEqual(created.toJson());
    await expect(client.createIssue(options)).rejects.toMatchObject({
      errorCode: 'RESOURCE_ALREADY_EXISTS',
    });
  });

  it('merges metadata, removes named keys, and supports sparse, clearing and no-op updates', async () => {
    const created = await client.createIssue({
      experimentId,
      name: 'Original issue',
      description: 'Original description',
      severity: IssueSeverity.LOW,
      categories: [IssueCategory.LATENCY],
      metadata: { keep: 'original', remove: 'old', 'app.owner': 'team-a' },
    });
    const updated = await client.updateIssue(created.issueId, {
      experimentId,
      name: 'Revised issue',
      severity: IssueSeverity.HIGH,
      status: IssueStatus.RESOLVED,
      metadata: { added: 'new', 'app.owner': 'team-b' },
      deleteMetadataKeys: ['remove'],
    });
    expect(updated).toMatchObject({
      name: 'Revised issue',
      description: created.description,
      categories: created.categories,
      status: IssueStatus.RESOLVED,
      severity: IssueSeverity.HIGH,
      metadata: { keep: 'original', added: 'new', 'app.owner': 'team-b' },
    });
    expect(updated.metadata).not.toHaveProperty('remove');

    const cleared = await client.updateIssue(created.issueId, {
      experimentId,
      description: '',
      categories: [],
      deleteMetadataKeys: ['app.owner'],
    });
    expect(cleared.description).toBe('');
    expect(cleared.categories ?? []).toEqual([]);
    expect(cleared.metadata).toEqual({ keep: 'original', added: 'new' });

    const unchanged = await client.updateIssue(created.issueId, { experimentId });
    expect(unchanged).toMatchObject({
      name: cleared.name,
      description: '',
      status: cleared.status,
      severity: cleared.severity,
      metadata: cleared.metadata,
    });
    expect((await client.getIssue(created.issueId, experimentId)).toJson()).toEqual(
      unchanged.toJson(),
    );
  });

  it('filters by status and source run using the OSS search grammar', async () => {
    const run = await makeRequest<{ run: { info: { run_id: string } } }>(
      'POST',
      `${TEST_TRACKING_URI}/api/2.0/mlflow/runs/create`,
      authProvider.getHeadersProvider(),
      { experiment_id: experimentId, start_time: Date.now() },
    );
    const sourceRunId = run.run.info.run_id;
    const sourced = await client.createIssue({
      experimentId,
      name: 'Run-detected issue',
      description: 'Detected in a run',
      severity: IssueSeverity.MEDIUM,
      sourceRunId,
    });
    const resolved = await client.createIssue({
      experimentId,
      name: 'Resolved issue',
      description: 'Already fixed',
      severity: IssueSeverity.HIGH,
      status: IssueStatus.RESOLVED,
      sourceRunId,
    });
    await client.createIssue({
      experimentId,
      name: 'Manual issue',
      description: 'No source run',
      severity: IssueSeverity.LOW,
    });
    expect(sourced.sourceRunId).toBe(sourceRunId);

    const pending = await client.searchIssues({
      experimentId,
      filter: `status = 'pending' AND source_run_id = '${sourceRunId}'`,
    });
    expect(pending.issues.map((issue) => issue.issueId)).toEqual([sourced.issueId]);

    const notPending = await client.searchIssues({
      experimentId,
      filter: `status != 'pending' AND source_run_id = '${sourceRunId}'`,
    });
    expect(notPending.issues.map((issue) => issue.issueId)).toEqual([resolved.issueId]);
  });

  it('paginates in timestamp order and returns full or basic views', async () => {
    const start = Date.now() - 10000;
    const issueIds: string[] = [];
    for (let index = 0; index < 3; index++) {
      const issue = await client.createIssue({
        experimentId,
        name: `Issue ${index}`,
        description: `Description ${index}`,
        severity: IssueSeverity.MEDIUM,
        metadata: { index: String(index) },
        lastSeenTimestamp: start + index * 1000,
      });
      issueIds.push(issue.issueId);
    }

    const foundIds: string[] = [];
    let pageToken: string | undefined;
    for (let page = 0; page < 3; page++) {
      const result = await client.searchIssues({
        experimentId,
        maxResults: 1,
        orderBy: ['last_seen_timestamp DESC'],
        pageToken,
        view: 'full',
      });
      expect(result.issues).toHaveLength(1);
      expect(result.issues[0].description).toBe(`Description ${2 - page}`);
      expect(result.issues[0].metadata).toEqual({ index: String(2 - page) });
      foundIds.push(result.issues[0].issueId);
      pageToken = result.nextPageToken;
      if (page < 2) {
        expect(pageToken).toEqual(expect.any(String));
      }
    }
    expect(foundIds).toEqual([...issueIds].reverse());

    const basic = await client.searchIssues({
      experimentId,
      orderBy: ['last_seen_timestamp ASC'],
      view: 'basic',
    });
    expect(basic.issues.map((issue) => issue.issueId)).toEqual(issueIds);
    for (const issue of basic.issues) {
      expect(issue.description).toBe('');
      expect(issue.metadata).toBeUndefined();
    }
  });

  it('returns empty searches and keeps experiments isolated', async () => {
    expect((await client.searchIssues({ experimentId })).issues).toEqual([]);
    const otherExperimentId = await client.createExperiment(`typescript-issues-${randomUUID()}`);
    try {
      const created = await client.createIssue({
        experimentId: otherExperimentId,
        name: 'Another experiment',
        description: 'Should not appear in the first experiment',
        severity: IssueSeverity.HIGH,
      });
      expect((await client.searchIssues({ experimentId })).issues).toEqual([]);
      expect(
        (
          await client.searchIssues({
            experimentId: otherExperimentId,
            filter: "status = 'rejected'",
          })
        ).issues,
      ).toEqual([]);
      expect(
        (await client.searchIssues({ experimentId: otherExperimentId })).issues.map(
          (issue) => issue.issueId,
        ),
      ).toEqual([created.issueId]);
    } finally {
      await client.deleteExperiment(otherExperimentId);
    }
  });
});
