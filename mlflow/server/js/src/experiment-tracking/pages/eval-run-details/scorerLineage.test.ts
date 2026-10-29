import { describe, expect, test } from '@jest/globals';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import {
  getConsistentScorerDigest,
  getConsistentScorerReference,
  parseRegisteredScorerReference,
} from './scorerLineage';

const registeredScorerMetadata = (scorerName: string, scorerVersion?: string): Record<string, string> => ({
  'mlflow.assessment.scorerName': scorerName,
  ...(scorerVersion === undefined ? {} : { 'mlflow.assessment.scorerVersion': scorerVersion }),
});

const traceWithAssessments = (assessments: { name: string; metadata?: Record<string, string> }[]) =>
  ({
    trace_id: 'tr-1',
    assessments: assessments.map(({ name, metadata }, index) => ({
      assessment_id: `a-${index}`,
      assessment_name: name,
      trace_id: 'tr-1',
      source: { source_type: 'LLM_JUDGE', source_id: name },
      create_time: '2026-01-01T00:00:00Z',
      last_update_time: '2026-01-01T00:00:00Z',
      feedback: { value: 'yes' },
      metadata,
    })),
  }) as unknown as ModelTraceInfoV3;

describe('parseRegisteredScorerReference', () => {
  test('reads the registered scorer name and version', () => {
    expect(parseRegisteredScorerReference(registeredScorerMetadata('quality/foo', '7'))).toEqual({
      scorerName: 'quality/foo',
      scorerVersion: 7,
    });
  });

  test('keeps the name when the version is missing or not a positive integer', () => {
    expect(parseRegisteredScorerReference(registeredScorerMetadata('quality'))).toEqual({
      scorerName: 'quality',
      scorerVersion: undefined,
    });
    expect(parseRegisteredScorerReference(registeredScorerMetadata('quality', '0'))?.scorerVersion).toBeUndefined();
    expect(
      parseRegisteredScorerReference(registeredScorerMetadata('quality', 'latest'))?.scorerVersion,
    ).toBeUndefined();
  });

  test.each<Record<string, string> | undefined>([
    undefined,
    {},
    { 'mlflow.assessment.scorerName': '' },
    { 'mlflow.assessment.scorerVersion': '2' },
  ])('finds no registered scorer in %p', (metadata) => {
    expect(parseRegisteredScorerReference(metadata)).toBeUndefined();
  });
});

describe('getConsistentScorerReference', () => {
  test('returns the scorer when every assessment with the name has the same one', () => {
    const traces = [
      traceWithAssessments([{ name: 'quality', metadata: registeredScorerMetadata('quality', '2') }]),
      traceWithAssessments([
        { name: 'quality', metadata: registeredScorerMetadata('quality', '2') },
        { name: 'safety' },
      ]),
    ];
    expect(getConsistentScorerReference(traces, 'quality')).toEqual({ scorerName: 'quality', scorerVersion: 2 });
  });

  test('returns nothing when versions differ or some assessments come from an ad-hoc scorer', () => {
    expect(
      getConsistentScorerReference(
        [
          traceWithAssessments([{ name: 'quality', metadata: registeredScorerMetadata('quality', '2') }]),
          traceWithAssessments([{ name: 'quality', metadata: registeredScorerMetadata('quality', '3') }]),
        ],
        'quality',
      ),
    ).toBeUndefined();
    expect(
      getConsistentScorerReference(
        [
          traceWithAssessments([{ name: 'quality', metadata: registeredScorerMetadata('quality', '2') }]),
          traceWithAssessments([{ name: 'quality' }]),
        ],
        'quality',
      ),
    ).toBeUndefined();
  });

  test('returns nothing when no assessment has the name', () => {
    expect(getConsistentScorerReference([traceWithAssessments([{ name: 'safety' }])], 'quality')).toBeUndefined();
    expect(getConsistentScorerReference([], 'quality')).toBeUndefined();
  });
});

describe('getConsistentScorerDigest', () => {
  const digestMetadata = (digest: string) => ({ 'mlflow.assessment.scorerDigest': digest });

  test('returns the digest when every assessment with the name reports the same one', () => {
    const traces = [
      traceWithAssessments([{ name: 'quality', metadata: digestMetadata('sha256:aaa') }, { name: 'tone' }]),
      traceWithAssessments([{ name: 'quality', metadata: digestMetadata('sha256:aaa') }]),
    ];
    expect(getConsistentScorerDigest(traces, 'quality')).toBe('sha256:aaa');
    expect(getConsistentScorerDigest(traces, 'tone')).toBeUndefined();
  });

  test('returns nothing when the digests differ or some assessments have none', () => {
    expect(
      getConsistentScorerDigest(
        [
          traceWithAssessments([{ name: 'quality', metadata: digestMetadata('sha256:aaa') }]),
          traceWithAssessments([{ name: 'quality', metadata: digestMetadata('sha256:bbb') }]),
        ],
        'quality',
      ),
    ).toBeUndefined();
    expect(
      getConsistentScorerDigest(
        [traceWithAssessments([{ name: 'quality', metadata: digestMetadata('sha256:aaa') }, { name: 'quality' }])],
        'quality',
      ),
    ).toBeUndefined();
  });
});
