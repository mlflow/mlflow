import { describe, expect, test } from '@jest/globals';

import { CREATE_CUSTOM_VIEW_QUERY_PARAM } from '@databricks/web-shared/model-trace-explorer';

import { getReviewQueueCreateCustomViewUrl } from './reviewQueueCustomViewNavigation';

describe('getReviewQueueCreateCustomViewUrl', () => {
  test('builds a hash-router Traces URL for the trace and preserves unrelated workspace context', () => {
    const url = getReviewQueueCreateCustomViewUrl({
      experimentId: 'experiment-1',
      traceId: 'tr-123',
      currentSearch:
        '?selectedQueueId=queue-1&selectedItemId=tr-123&startReview=true&workspace=team-ws&spanId=old&traceId=stale',
    });

    const [path, query] = url.split('?');
    expect(path).toBe('/#/experiments/experiment-1/traces');
    const params = new URLSearchParams(query);
    expect(params.get('selectedEvaluationId')).toBe('tr-123');
    expect(params.get(CREATE_CUSTOM_VIEW_QUERY_PARAM)).toBe('true');
    expect(params.get('workspace')).toBe('team-ws');
    expect(params.has('selectedQueueId')).toBe(false);
    expect(params.has('selectedItemId')).toBe(false);
    expect(params.has('startReview')).toBe(false);
    expect(params.has('spanId')).toBe(false);
    expect(params.has('traceId')).toBe(false);
  });
});
