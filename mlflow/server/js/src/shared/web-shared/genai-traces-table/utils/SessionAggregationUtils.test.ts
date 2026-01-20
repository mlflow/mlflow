import { describe, it, expect } from '@jest/globals';

import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import { ASSESSMENT_SESSION_METADATA_KEY } from '../../model-trace-explorer/constants';
import type { AssessmentInfo } from '../types';

import { aggregatePassFailAssessments } from './SessionAggregationUtils';

// Helper to create a minimal AssessmentInfo for testing
const createAssessmentInfo = (overrides: Partial<AssessmentInfo> = {}): AssessmentInfo => ({
  name: 'test_assessment',
  displayName: 'Test Assessment',
  isKnown: false,
  isOverall: false,
  metricName: 'test_metric',
  isCustomMetric: false,
  isEditable: false,
  isRetrievalAssessment: false,
  dtype: 'pass-fail',
  uniqueValues: new Set(),
  docsLink: '',
  missingTooltip: '',
  description: '',
  ...overrides,
});

// Helper to create a minimal trace with assessments
const createTrace = (assessments: any[] = []): ModelTraceInfoV3 =>
  ({
    trace_id: 'test-trace-id',
    assessments,
  }) as ModelTraceInfoV3;

describe('SessionAggregationUtils', () => {
  describe('aggregatePassFailAssessments', () => {
    const assessmentInfo = createAssessmentInfo({ name: 'correctness', dtype: 'pass-fail' });

    it('returns zero counts for empty traces array', () => {
      const result = aggregatePassFailAssessments([], assessmentInfo);
      expect(result).toEqual({ passCount: 0, totalCount: 0 });
    });

    it('counts all passing assessments correctly', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      expect(result).toEqual({ passCount: 2, totalCount: 2 });
    });

    it('counts all failing assessments correctly', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'no' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'no' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      expect(result).toEqual({ passCount: 0, totalCount: 2 });
    });

    it('counts mixed pass/fail assessments correctly', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'no' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'no' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      expect(result).toEqual({ passCount: 2, totalCount: 4 });
    });

    it('skips traces without the target assessment', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'other_assessment',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      expect(result).toEqual({ passCount: 1, totalCount: 1 });
    });

    it('skips session-level assessments', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
            metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'true' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      expect(result).toEqual({ passCount: 1, totalCount: 1 });
    });

    it('skips assessments without feedback', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            // No feedback property
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      expect(result).toEqual({ passCount: 1, totalCount: 1 });
    });

    it('works with boolean dtype', () => {
      const booleanAssessmentInfo = createAssessmentInfo({ name: 'is_correct', dtype: 'boolean' });
      const traces = [
        createTrace([
          {
            assessment_name: 'is_correct',
            feedback: { value: true },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'is_correct',
            feedback: { value: false },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'is_correct',
            feedback: { value: true },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, booleanAssessmentInfo);
      expect(result).toEqual({ passCount: 2, totalCount: 3 });
    });

    it('counts multiple assessments with the same name per trace', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'judge1' },
          },
          {
            assessment_name: 'correctness',
            feedback: { value: 'no' },
            source: { source_type: 'HUMAN', source_id: 'human1' },
          },
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'judge2' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'judge1' },
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      // 3 assessments in first trace + 1 in second = 4 total
      // 2 pass in first trace + 1 pass in second = 3 pass
      expect(result).toEqual({ passCount: 3, totalCount: 4 });
    });

    it('filters out invalid assessments (valid === false)', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
            valid: false, // Should be filtered out
          },
          {
            assessment_name: 'correctness',
            feedback: { value: 'yes' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
            valid: true, // Should be included
          },
        ]),
        createTrace([
          {
            assessment_name: 'correctness',
            feedback: { value: 'no' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
            // valid is undefined, should be included
          },
        ]),
      ];

      const result = aggregatePassFailAssessments(traces, assessmentInfo);
      // Only 2 valid assessments: 1 pass, 1 fail
      expect(result).toEqual({ passCount: 1, totalCount: 2 });
    });
  });
});
