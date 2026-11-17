import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { FilterOp } from '@databricks/web-shared/traces-table';
import * as FeatureUtils from '../../../common/utils/FeatureUtils';
import { MlflowService } from '../../sdk/MlflowService';
import { MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG } from '../../constants';
import {
  compileIssueDetectionTraceScope,
  readIssueDetectionTraceScope,
  saveIssueDetectionTraceScope,
} from './issueDetectionTraceScope';

jest.mock('../../sdk/MlflowService', () => ({ MlflowService: { setExperimentTag: jest.fn() } }));

describe('shared issue detection trace scope', () => {
  beforeEach(() => {
    jest.mocked(MlflowService.setExperimentTag).mockReset();
  });

  test('legacy schedule filters become editable setup filters without changing their meaning', () => {
    expect(
      readIssueDetectionTraceScope([
        {
          key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
          value: JSON.stringify([
            { key: ' env ', value: 'prod' },
            { field: 'tag', operator: '=', key: 'team', value: 'support' },
          ]),
        },
      ]),
    ).toMatchObject({
      searchQuery: '',
      filters: [
        { field: 'metadata', operator: '=', key: 'env', value: 'prod' },
        { field: 'tag', operator: '=', key: 'team', value: 'support' },
      ],
    });
  });

  test('rich filters and text round trip and share the scheduled-run expression', async () => {
    const scope = {
      searchQuery: 'feedback',
      filters: [{ field: 'duration', operator: FilterOp.GREATER_THAN, value: '1000' }],
    };
    await saveIssueDetectionTraceScope('1', scope);
    const tag = jest.mocked(MlflowService.setExperimentTag).mock.calls[0][0];
    expect(tag.key).toBe(MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG);
    expect(readIssueDetectionTraceScope([tag])).toMatchObject(scope);
    expect(
      JSON.parse(tag.value)
        .map((entry: { compiled: string }) => entry.compiled)
        .join(' AND '),
    ).toBe(compileIssueDetectionTraceScope(scope));
    expect(compileIssueDetectionTraceScope(scope)).toBe(
      "attributes.execution_time_ms > 1000 AND trace.text ILIKE '%feedback%'",
    );
  });

  test('an exact trace search still intersects the other saved constraints', () => {
    expect(
      compileIssueDetectionTraceScope({
        searchQuery: 'tr-0123456789abcdef0123456789abcdef',
        filters: [{ field: 'duration', operator: FilterOp.GREATER_THAN, value: '1000' }],
      }),
    ).toContain(
      "attributes.execution_time_ms > 1000 AND attributes.request_id = 'tr-0123456789abcdef0123456789abcdef'",
    );
  });

  test('an explicit reset clears the persisted scope', async () => {
    await saveIssueDetectionTraceScope('1', { searchQuery: '   ', filters: [] });
    expect(MlflowService.setExperimentTag).toHaveBeenCalledWith({
      experiment_id: '1',
      key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
      value: '',
    });
  });

  test('incomplete filters are rejected before overwriting saved scope', async () => {
    await expect(
      saveIssueDetectionTraceScope('1', {
        searchQuery: '',
        filters: [{ field: 'tag', operator: FilterOp.EQUALS, value: 'prod' }],
      }),
    ).rejects.toThrow('Complete or remove');
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
  });
});

test.each(['not-json', '{}', '[{"field":"future","operator":"=","value":"x"}]'])(
  'an unreadable active scope fails closed: %s',
  (value) => {
    expect(() =>
      readIssueDetectionTraceScope([{ key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG, value }]),
    ).toThrow();
  },
);

test('saved compiled scope stays identical when filter feature flags change', async () => {
  const flag = jest.spyOn(FeatureUtils, 'shouldEnableNewEvalRunsUi').mockReturnValue(true);
  try {
    await saveIssueDetectionTraceScope('1', {
      searchQuery: '',
      filters: [{ field: 'state', operator: FilterOp.NOT_EQUALS, value: 'ERROR' }],
    });
    const tag = jest.mocked(MlflowService.setExperimentTag).mock.calls.slice(-1)[0][0];
    const scheduled = JSON.parse(tag.value)
      .map((entry: { compiled: string }) => entry.compiled)
      .join(' AND ');
    flag.mockReturnValue(false);
    expect(compileIssueDetectionTraceScope(readIssueDetectionTraceScope([tag]))).toBe(scheduled);
    expect(scheduled).toBe("attributes.status != 'ERROR'");
  } finally {
    flag.mockRestore();
  }
});

test.each([
  { searchQuery: 'x'.repeat(10000), filters: [] },
  { searchQuery: '', filters: [{ field: 'tag', key: 'bad`key', operator: FilterOp.EQUALS, value: 'value' }] },
  { searchQuery: "don't", filters: [] },
  { searchQuery: '', filters: [{ field: 'tag', key: 'team', operator: FilterOp.EQUALS, value: "don't" }] },
])('unusable saved expressions do not overwrite scope: %j', async (scope) => {
  jest.mocked(MlflowService.setExperimentTag).mockClear();
  await expect(saveIssueDetectionTraceScope('1', scope)).rejects.toThrow();
  expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
});
