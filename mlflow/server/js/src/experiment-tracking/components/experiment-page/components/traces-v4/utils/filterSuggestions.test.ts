import { describe, expect, test } from '@jest/globals';
import { NOTES_ASSESSMENT_NAME } from '@databricks/web-shared/model-trace-explorer/assessments-pane/AssessmentsPaneNotesSection';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { buildTraceFilterSuggestions, getTraceFilterValueOptions } from './filterSuggestions';

describe('buildTraceFilterSuggestions', () => {
  test('derives sorted distinct loaded values, including previews and run IDs', () => {
    const suggestions = buildTraceFilterSuggestions([
      makeTrace('trace-b', {
        request: 'full-input-b',
        request_preview: 'preview-input-b',
        response: 'full-output-b',
        response_preview: 'preview-output-b',
        tags: { 'mlflow.traceName': 'zeta' },
        trace_metadata: {
          'mlflow.trace.user': 'user-b',
          'mlflow.trace.session': 'session-b',
          'mlflow.sourceRun': 'run-b',
          'mlflow.source.name': 'source-b',
        },
      }),
      makeTrace('trace-a', {
        request: 'full-input-a',
        request_preview: 'preview-input-a',
        response: 'full-output-a',
        response_preview: 'preview-output-a',
        tags: { 'mlflow.traceName': 'alpha' },
        trace_metadata: {
          'mlflow.trace.user': 'user-a',
          'mlflow.trace.session': 'session-a',
          'mlflow.sourceRun': 'run-a',
          'mlflow.source.name': 'source-a',
        },
      }),
    ]);

    expect(suggestions.values.trace_name).toEqual([
      { value: 'alpha', label: 'alpha' },
      { value: 'zeta', label: 'zeta' },
    ]);
    expect(suggestions.values.user).toEqual([
      { value: 'user-a', label: 'user-a' },
      { value: 'user-b', label: 'user-b' },
    ]);
    expect(suggestions.values.session).toEqual([
      { value: 'session-a', label: 'session-a' },
      { value: 'session-b', label: 'session-b' },
    ]);
    expect(suggestions.values.run_name).toEqual([
      { value: 'run-a', label: 'run-a' },
      { value: 'run-b', label: 'run-b' },
    ]);
    expect(suggestions.values.source).toEqual([
      { value: 'source-a', label: 'source-a' },
      { value: 'source-b', label: 'source-b' },
    ]);
    expect(suggestions.values.input).toEqual([
      { value: 'full-input-a', label: 'full-input-a' },
      { value: 'full-input-b', label: 'full-input-b' },
      { value: 'preview-input-a', label: 'preview-input-a' },
      { value: 'preview-input-b', label: 'preview-input-b' },
    ]);
    expect(suggestions.values.output.map(({ value }) => value)).toEqual([
      'full-output-a',
      'full-output-b',
      'preview-output-a',
      'preview-output-b',
    ]);
  });

  test('groups tag, metadata, and displayable assessment values by key', () => {
    const suggestions = buildTraceFilterSuggestions([
      makeTrace('trace-1', {
        tags: { env: 'prod', team: 'ml' },
        trace_metadata: { owner: 'alice', region: 'us-east' },
        assessments: [
          makeFeedbackAssessment('quality', false),
          makeFeedbackAssessment('score', 4),
          makeFeedbackAssessment('invalid', 'skip', { valid: false }),
          makeFeedbackAssessment('session-quality', 'Pass', {
            metadata: { 'mlflow.trace.session': 'session-1' },
          }),
          makeFeedbackAssessment(NOTES_ASSESSMENT_NAME, 'hidden note'),
        ],
      }),
      makeTrace('trace-2', {
        tags: { env: 'staging', team: 'ml' },
        trace_metadata: { owner: 'bob', region: 'us-west' },
        assessments: [makeFeedbackAssessment('quality', false), makeFeedbackAssessment('score', 2.5)],
      }),
    ]);

    expect(suggestions.keyedValues.tag.keys.map(({ value }) => value)).toEqual(['env', 'team']);
    expect(getTraceFilterValueOptions(suggestions, 'tag', 'env')).toEqual([
      { value: 'prod', label: 'prod' },
      { value: 'staging', label: 'staging' },
    ]);
    expect(suggestions.keyedValues.metadata.keys.map(({ value }) => value)).toEqual(['owner', 'region']);
    expect(getTraceFilterValueOptions(suggestions, 'metadata', 'owner').map(({ value }) => value)).toEqual([
      'alice',
      'bob',
    ]);
    // Invalid, session-level and internal assessments get no column, so they aren't suggested either.
    expect(suggestions.assessmentNames.map(({ value }) => value)).toEqual(['quality', 'score']);
    expect(getTraceFilterValueOptions(suggestions, 'assessment', 'quality')).toEqual([
      { value: 'false', label: 'false' },
    ]);
    expect(getTraceFilterValueOptions(suggestions, 'assessment', 'score').map(({ value }) => value)).toEqual([
      '2.5',
      '4',
    ]);
    expect(getTraceFilterValueOptions(suggestions, 'assessment', 'invalid')).toEqual([]);
    expect(getTraceFilterValueOptions(suggestions, 'tag', 'missing')).toEqual([]);
  });

  test('keeps expectations apart from feedback', () => {
    const suggestions = buildTraceFilterSuggestions([
      makeTrace('trace-1', {
        assessments: [
          makeFeedbackAssessment('correctness', 'yes'),
          {
            ...makeFeedbackAssessment('expected_response', 'x'),
            feedback: undefined,
            expectation: { value: 'Settings' },
          } as any,
        ],
      }),
    ]);

    expect(suggestions.assessmentNames.map(({ value }) => value)).toEqual(['correctness']);
    expect(suggestions.keyedValues.expectation.keys.map(({ value }) => value)).toEqual(['expected_response']);
    expect(getTraceFilterValueOptions(suggestions, 'expectation', 'expected_response')).toEqual([
      { value: 'Settings', label: 'Settings' },
    ]);
  });
});
