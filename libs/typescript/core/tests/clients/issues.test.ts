import { MlflowClient } from '../../src/clients/client';
import { MlflowHttpError } from '../../src/clients/utils';
import { Issue, IssueCategory, IssueSeverity, IssueStatus } from '../../src';

describe('MlflowClient issues', () => {
  const host = 'https://tracking.example.com';
  const experimentId = '123';
  const issueId = 'iss-0123456789abcdef0123456789abcdef';
  const timestamp = 1735689600123;
  const ossIssue = {
    issue_id: issueId,
    experiment_id: experimentId,
    name: 'Incorrect answer',
    description: 'The response contradicts the source.',
    status: 'pending',
    severity: 'high',
    categories: ['correctness', 'custom-category'],
    created_timestamp: String(timestamp),
    last_updated_timestamp: String(timestamp + 1000),
    last_seen_timestamp: String(timestamp),
    metadata: { owner: 'quality' },
  };
  const databricksIssue = {
    issue_id: issueId,
    title: ossIssue.name,
    description: ossIssue.description,
    status: 'PENDING',
    severity: 'HIGH',
    categories: ossIssue.categories,
    create_time: new Date(timestamp).toISOString(),
    update_time: new Date(timestamp + 1000).toISOString(),
    last_seen_time: new Date(timestamp).toISOString(),
    created_by: 'creator',
    last_updated_by: 'editor',
    source_type: 'MANUAL',
    source_id: 'manual-source',
    metadata: ossIssue.metadata,
  };
  let fetchMock: jest.Mock;
  const originalFetch = global.fetch;

  function createClient(databricks = false): MlflowClient {
    return new MlflowClient({
      trackingUri: databricks ? 'databricks://test-profile' : host,
      authProvider: {
        getHost: () => host,
        getHeadersProvider: () => () =>
          Promise.resolve({ 'Content-Type': 'application/json', Authorization: 'Bearer test' }),
        getDatabricksToken: () => undefined,
      },
    });
  }

  function respond(body: unknown): void {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(body)));
  }

  function request(): { url: URL; init: RequestInit; body: Record<string, unknown> } {
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    return {
      url: new URL(url),
      init,
      body: init.body ? (JSON.parse(init.body as string) as Record<string, unknown>) : {},
    };
  }

  beforeEach(() => {
    fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    jest.restoreAllMocks();
  });

  describe.each([false, true])('request validation (Databricks: %s)', (databricks) => {
    it('requires an experiment for each operation', async () => {
      const client = createClient(databricks);
      await expect(
        client.createIssue({
          experimentId: '',
          name: ossIssue.name,
          description: ossIssue.description,
          severity: IssueSeverity.HIGH,
        }),
      ).rejects.toThrow();
      await expect(client.getIssue(issueId, '')).rejects.toThrow();
      await expect(client.updateIssue(issueId, { experimentId: '' })).rejects.toThrow();
      await expect(client.searchIssues({ experimentId: '' })).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it.each(['', undefined])('requires an issue ID for get and update: %s', async (id) => {
      const client = createClient(databricks);
      await expect(client.getIssue(id as unknown as string, experimentId)).rejects.toThrow();
      await expect(client.updateIssue(id as unknown as string, { experimentId })).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it.each([NaN, -1, 0.5, Infinity, Number.MAX_SAFE_INTEGER])(
      'rejects an invalid observation timestamp: %s',
      async (lastSeenTimestamp) => {
        await expect(
          createClient(databricks).updateIssue(issueId, { experimentId, lastSeenTimestamp }),
        ).rejects.toThrow();
        expect(fetchMock).not.toHaveBeenCalled();
      },
    );

    it('rejects metadata keys that are both updated and deleted', async () => {
      await expect(
        createClient(databricks).updateIssue(issueId, {
          experimentId,
          metadata: { owner: 'quality' },
          deleteMetadataKeys: ['owner'],
        }),
      ).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  it('round-trips the exported Issue entity without losing metadata or zero timestamps', () => {
    const issue = Issue.fromJson({
      ...ossIssue,
      status: IssueStatus.PENDING,
      severity: IssueSeverity.HIGH,
      created_timestamp: '0',
      last_updated_timestamp: '0',
      last_seen_timestamp: '0',
      trace_count: 0,
      source_run_id: 'run-1',
      created_by: 'creator',
      last_updated_by: 'editor',
      metadata: { owner: 'quality', 'app.owner': 'team-a' },
    });
    expect(issue.toJson()).toMatchObject({
      issue_id: issueId,
      experiment_id: experimentId,
      categories: [IssueCategory.CORRECTNESS, 'custom-category'],
      created_timestamp: 0,
      last_updated_timestamp: 0,
      last_seen_timestamp: 0,
      trace_count: 0,
      source_run_id: 'run-1',
      created_by: 'creator',
      last_updated_by: 'editor',
      metadata: { owner: 'quality', 'app.owner': 'team-a' },
    });
    expect(Issue.fromJson(issue.toJson())).toEqual(issue);
  });

  describe('OSS', () => {
    it('creates an issue using the existing authenticated JSON transport', async () => {
      respond({ issue: ossIssue });
      const issue = await createClient().createIssue({
        experimentId,
        name: ossIssue.name,
        description: ossIssue.description,
        severity: IssueSeverity.HIGH,
        categories: [IssueCategory.CORRECTNESS, 'custom-category'],
        metadata: { owner: 'quality' },
        issueId,
        lastSeenTimestamp: timestamp,
        sourceRunId: 'run-1',
      });

      expect(request().url.pathname).toBe('/api/3.0/mlflow/issues');
      expect(request().init.method).toBe('POST');
      expect(request().init.headers).toMatchObject({ Authorization: 'Bearer test' });
      expect(request().body).toMatchObject({
        experiment_id: experimentId,
        name: ossIssue.name,
        description: ossIssue.description,
        severity: 'high',
        status: 'pending',
        categories: ossIssue.categories,
        metadata: ossIssue.metadata,
        issue_id: issueId,
        last_seen_timestamp: timestamp,
        source_run_id: 'run-1',
      });
      expect(issue).toBeInstanceOf(Issue);
      expect(issue).toMatchObject({
        issueId,
        experimentId,
        name: ossIssue.name,
        status: IssueStatus.PENDING,
        severity: IssueSeverity.HIGH,
        categories: [IssueCategory.CORRECTNESS, 'custom-category'],
        createdTimestamp: timestamp,
        lastUpdatedTimestamp: timestamp + 1000,
        lastSeenTimestamp: timestamp,
      });
    });

    it('encodes the issue ID in GET and uses the OSS wrapper', async () => {
      respond({ issue: ossIssue });
      const issue = await createClient().getIssue('iss-a/b ?#', experimentId);
      expect(request().url.pathname).toBe('/api/3.0/mlflow/issues/iss-a%2Fb%20%3F%23');
      expect(request().init.method).toBe('GET');
      expect(request().init.body).toBeUndefined();
      expect(issue.issueId).toBe(issueId);
    });

    it('updates only supplied fields and preserves explicit clears', async () => {
      respond({ issue: { ...ossIssue, description: '', categories: [] } });
      await createClient().updateIssue(issueId, {
        experimentId,
        description: '',
        categories: [],
        metadata: { 'owner.name': 'new-owner' },
        deleteMetadataKeys: ['old-owner'],
        lastSeenTimestamp: 0,
      });

      expect(request().url.pathname).toBe(`/api/3.0/mlflow/issues/${issueId}`);
      expect(request().init.method).toBe('PATCH');
      expect(request().body).toMatchObject({
        description: '',
        categories: [],
        metadata: { 'owner.name': 'new-owner' },
        delete_metadata_keys: ['old-owner'],
        last_seen_timestamp: 0,
      });
      expect(request().body).not.toHaveProperty('name');
      expect(request().body).not.toHaveProperty('status');
      expect(request().body).not.toHaveProperty('severity');
    });

    it('passes search filters, order and cursor without inventing separate filters', async () => {
      respond({ issues: [ossIssue], next_page_token: 'next+/=', total_count: '4' });
      const result = await createClient().searchIssues({
        experimentId,
        filter: "status != 'rejected' AND source_run_id = 'run-1'",
        maxResults: 2,
        orderBy: ['last_seen_timestamp DESC'],
        pageToken: 'previous+/=',
        view: 'basic',
      });

      expect(request().url.pathname).toBe('/api/3.0/mlflow/issues/search');
      expect(request().body).toMatchObject({
        experiment_id: experimentId,
        filter_string: "status != 'rejected' AND source_run_id = 'run-1'",
        max_results: 2,
        order_by: ['last_seen_timestamp DESC'],
        page_token: 'previous+/=',
        view: 'basic',
      });
      expect(result.issues[0]).toBeInstanceOf(Issue);
      expect(result.nextPageToken).toBe('next+/=');
      expect(result.totalCount).toBe(4);
    });

    it('handles an empty search response', async () => {
      respond({});
      expect(await createClient().searchIssues({ experimentId })).toMatchObject({ issues: [] });
    });

    it('preserves backend error information', async () => {
      fetchMock.mockResolvedValue(
        new Response(
          JSON.stringify({ error_code: 'RESOURCE_DOES_NOT_EXIST', message: 'Issue missing' }),
          { status: 404, statusText: 'Not Found' },
        ),
      );
      await expect(createClient().getIssue(issueId, experimentId)).rejects.toMatchObject({
        name: 'MlflowHttpError',
        status: 404,
        errorCode: 'RESOURCE_DOES_NOT_EXIST',
      });
      await expect(createClient().getIssue(issueId, experimentId)).rejects.toBeInstanceOf(
        MlflowHttpError,
      );
    });
  });

  describe('Databricks', () => {
    it('routes by tracking URI and adapts caller ID, enums and timestamps', async () => {
      respond(databricksIssue);
      const issue = await createClient(true).createIssue({
        experimentId: 'exp/with space',
        name: ossIssue.name,
        description: '',
        severity: IssueSeverity.HIGH,
        status: IssueStatus.RESOLVED,
        categories: [IssueCategory.CORRECTNESS, 'custom-category'],
        issueId,
        lastSeenTimestamp: 0,
      });

      expect(request().url.pathname).toBe('/api/2.0/mlflow/experiments/exp%2Fwith%20space/issues');
      expect(request().url.searchParams.get('issue_id')).toBe(issueId);
      expect(request().body).toMatchObject({
        title: ossIssue.name,
        description: '',
        severity: 'HIGH',
        status: 'RESOLVED',
        categories: ossIssue.categories,
        source_type: 'MANUAL',
      });
      expect(Date.parse(request().body.last_seen_time as string)).toBe(0);
      expect(request().body).not.toHaveProperty('experiment_id');
      expect(request().body).not.toHaveProperty('issue_id');
      expect(issue).toMatchObject({
        issueId,
        experimentId: 'exp/with space',
        name: ossIssue.name,
        severity: IssueSeverity.HIGH,
        status: IssueStatus.PENDING,
        createdTimestamp: timestamp,
        lastUpdatedTimestamp: timestamp + 1000,
        lastSeenTimestamp: timestamp,
        createdBy: 'creator',
        lastUpdatedBy: 'editor',
      });
      expect(issue.sourceRunId).toBeUndefined();
    });

    it('uses ADHOC_RUN and a current last-seen timestamp for run-created issues', async () => {
      jest.spyOn(Date, 'now').mockReturnValue(timestamp);
      respond({ ...databricksIssue, source_type: 'ADHOC_RUN', source_id: 'run-1' });
      const issue = await createClient(true).createIssue({
        experimentId,
        name: ossIssue.name,
        description: ossIssue.description,
        severity: IssueSeverity.LOW,
        sourceRunId: 'run-1',
      });
      expect(request().body).toMatchObject({
        source_type: 'ADHOC_RUN',
        source_id: 'run-1',
      });
      expect(Date.parse(request().body.last_seen_time as string)).toBe(timestamp);
      expect(issue.sourceRunId).toBe('run-1');
    });

    it('gets an issue from encoded experiment and issue paths', async () => {
      respond({ ...databricksIssue, source_type: 'SCHEDULED_RUN', source_id: 'run-2' });
      const issue = await createClient(true).getIssue('iss-a/b ?', 'exp/a ?');
      expect(request().url.pathname).toBe(
        '/api/2.0/mlflow/experiments/exp%2Fa%20%3F/issues/iss-a%2Fb%20%3F',
      );
      expect(issue.sourceRunId).toBe('run-2');
    });

    it('omits unspecified severity and basic-view description safely', async () => {
      respond({
        ...databricksIssue,
        description: undefined,
        severity: 'ISSUE_SEVERITY_UNSPECIFIED',
        last_seen_time: undefined,
      });
      const issue = await createClient(true).getIssue(issueId, experimentId);
      expect(issue.description).toBe('');
      expect(issue.severity).toBeUndefined();
      expect(issue.lastSeenTimestamp).toBeUndefined();
    });

    it('uses individual metadata masks and keeps explicit clears', async () => {
      respond(databricksIssue);
      await createClient(true).updateIssue(issueId, {
        experimentId,
        description: '',
        categories: [],
        status: IssueStatus.REJECTED,
        metadata: { owner: 'new-owner', 'app.owner': 'new-team' },
        deleteMetadataKeys: ['old.owner'],
        lastSeenTimestamp: 0,
      });
      expect(request().init.method).toBe('PATCH');
      expect(request().url.searchParams.get('update_mask')?.split(',')).toEqual(
        expect.arrayContaining([
          'description',
          'categories',
          'status',
          'last_seen_time',
          'metadata.owner',
          'metadata.`app.owner`',
          'metadata.`old.owner`',
        ]),
      );
      expect(request().url.searchParams.get('update_mask')?.split(',')).toHaveLength(7);
      expect(request().body).toMatchObject({
        name: `experiments/${experimentId}/issues/${issueId}`,
        description: '',
        categories: [],
        status: 'REJECTED',
        metadata: { owner: 'new-owner', 'app.owner': 'new-team' },
      });
      expect(Date.parse(request().body.last_seen_time as string)).toBe(0);
      expect(request().body).not.toHaveProperty('title');
      expect(request().body).not.toHaveProperty('severity');
      expect(request().body).not.toHaveProperty('delete_metadata_keys');
    });

    it('maps update names to title instead of overwriting the resource name', async () => {
      respond(databricksIssue);
      await createClient(true).updateIssue(issueId, {
        experimentId,
        name: 'New title',
        severity: IssueSeverity.MEDIUM,
      });
      expect(request().body).toMatchObject({
        name: `experiments/${experimentId}/issues/${issueId}`,
        title: 'New title',
        severity: 'MEDIUM',
      });
      expect(request().url.searchParams.get('update_mask')?.split(',')).toEqual(
        expect.arrayContaining(['title', 'severity']),
      );
    });

    it('gets the existing issue for a no-op update', async () => {
      respond(databricksIssue);
      await createClient(true).updateIssue(issueId, {
        experimentId,
        metadata: {},
        deleteMetadataKeys: [],
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(request().init.method).toBe('GET');
      expect(request().url.search).toBe('');
    });

    it('translates search filters, ordering, view and pagination', async () => {
      respond({ issues: [databricksIssue], next_page_token: 'next+/=', total_size: '12' });
      const result = await createClient(true).searchIssues({
        experimentId,
        filter: "status != 'rejected' AND status = 'pending'",
        maxResults: 3,
        pageToken: 'previous+/=',
        orderBy: ['last_updated_timestamp DESC'],
        view: 'basic',
      });
      expect(request().url.pathname).toBe(
        `/api/2.0/mlflow/experiments/${experimentId}/issues:search`,
      );
      expect(request().body).toMatchObject({
        parent: `experiments/${experimentId}`,
        statuses: ['PENDING'],
        page_size: 3,
        page_token: 'previous+/=',
        sort_key: 'UPDATE_TIME',
        sort_order: 'DESCENDING',
        view: 'BASIC',
      });
      expect(request().body).not.toHaveProperty('filter_string');
      expect(result.issues[0].name).toBe(ossIssue.name);
      expect(result.nextPageToken).toBe('next+/=');
      expect(result.totalCount).toBe(12);
    });

    it('requests full issue descriptions by default', async () => {
      respond({ issues: [] });
      await createClient(true).searchIssues({ experimentId });
      expect(request().body.view).toBe('FULL');
    });

    it('trims status values inside quotes consistently with OSS', async () => {
      respond({ issues: [] });
      await createClient(true).searchIssues({ experimentId, filter: "status = ' pending '" });
      expect(request().body.statuses).toEqual(['PENDING']);
    });

    it.each([
      ['severity', 'SEVERITY'],
      ['last_seen_timestamp', 'LAST_SEEN_TIME'],
      ['last_updated_timestamp', 'UPDATE_TIME'],
      ['created_timestamp', 'CREATE_TIME'],
    ])('translates ascending ordering on %s', async (field, sortKey) => {
      respond({});
      await createClient(true).searchIssues({ experimentId, orderBy: [field] });
      expect(request().body).toMatchObject({ sort_key: sortKey, sort_order: 'ASCENDING' });
    });

    it.each([
      "status = 'pending' AND status = 'resolved'",
      "status != 'pending' AND status != 'resolved' AND status != 'rejected'",
    ])('returns no issues for contradictory status predicates: %s', async (filter) => {
      const result = await createClient(true).searchIssues({ experimentId, filter });
      expect(result.issues).toEqual([]);
      expect(result.nextPageToken).toBeUndefined();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it.each([
      "source_run_id = 'run-1'",
      "status = 'pending' OR status = 'resolved'",
      "status = 'pending' AND severity = 'high'",
      "status = 'unknown'",
      "status = 'pending' garbage",
    ])('rejects unsupported filters instead of searching unfiltered: %s', async (filter) => {
      await expect(createClient(true).searchIssues({ experimentId, filter })).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it.each([['name ASC'], ['severity DESC', 'created_timestamp DESC']])(
      'rejects unsupported Databricks ordering: %s',
      async (...orderBy) => {
        await expect(createClient(true).searchIssues({ experimentId, orderBy })).rejects.toThrow();
        expect(fetchMock).not.toHaveBeenCalled();
      },
    );

    it('rejects an unsupported Databricks severity before creating an issue', async () => {
      await expect(
        createClient(true).createIssue({
          experimentId,
          name: ossIssue.name,
          description: ossIssue.description,
          severity: IssueSeverity.NOT_AN_ISSUE,
        }),
      ).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('rejects an unsupported Databricks severity before updating an issue', async () => {
      await expect(
        createClient(true).updateIssue(issueId, {
          experimentId,
          severity: IssueSeverity.NOT_AN_ISSUE,
        }),
      ).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it.each(['comma,key', 'back`tick'])(
      'rejects unrepresentable metadata mask keys: %s',
      async (key) => {
        await expect(
          createClient(true).updateIssue(issueId, { experimentId, deleteMetadataKeys: [key] }),
        ).rejects.toThrow();
        expect(fetchMock).not.toHaveBeenCalled();
      },
    );
  });
});
