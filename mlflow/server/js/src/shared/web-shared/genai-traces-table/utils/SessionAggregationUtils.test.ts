import { describe, it, expect } from '@jest/globals';

import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import { ASSESSMENT_SESSION_METADATA_KEY } from '../../model-trace-explorer/constants';
import type { AssessmentInfo } from '../types';

import {
  aggregateNumericAssessments,
  aggregatePassFailAssessments,
  aggregateStringAssessments,
} from './SessionAggregationUtils';

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

  describe('aggregateNumericAssessments', () => {
    it('returns null average for empty traces array', () => {
      const result = aggregateNumericAssessments([], 'numeric_assessment');
      expect(result).toEqual({ average: null, count: 0 });
    });

    it('calculates average correctly', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'score',
            feedback: { value: 0.8 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'score',
            feedback: { value: 0.6 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'score',
            feedback: { value: 1.0 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregateNumericAssessments(traces, 'score');
      expect(result.count).toBe(3);
      expect(result.average).toBeCloseTo(0.8, 5);
    });

    it('skips non-numeric values', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'score',
            feedback: { value: 0.5 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'score',
            feedback: { value: 'not a number' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'score',
            feedback: { value: 1.5 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregateNumericAssessments(traces, 'score');
      expect(result.count).toBe(2);
      expect(result.average).toBe(1.0);
    });

    it('skips invalid assessments', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'score',
            feedback: { value: 10 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
            valid: false,
          },
          {
            assessment_name: 'score',
            feedback: { value: 20 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregateNumericAssessments(traces, 'score');
      expect(result).toEqual({ average: 20, count: 1 });
    });
  });

  describe('aggregateStringAssessments', () => {
    it('returns empty map for empty traces array', () => {
      const result = aggregateStringAssessments([], 'string_assessment');
      expect(result.valueCounts.size).toBe(0);
      expect(result.totalCount).toBe(0);
    });

    it('counts unique string values correctly', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 'good' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 'bad' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 'good' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 'neutral' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregateStringAssessments(traces, 'category');
      expect(result.totalCount).toBe(4);
      expect(result.valueCounts.size).toBe(3);
      expect(result.valueCounts.get('good')).toBe(2);
      expect(result.valueCounts.get('bad')).toBe(1);
      expect(result.valueCounts.get('neutral')).toBe(1);
    });

    it('skips non-string values', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 'valid' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 123 },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: true },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregateStringAssessments(traces, 'category');
      expect(result.totalCount).toBe(1);
      expect(result.valueCounts.size).toBe(1);
      expect(result.valueCounts.get('valid')).toBe(1);
    });

    it('skips invalid assessments', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 'invalid' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
            valid: false,
          },
          {
            assessment_name: 'category',
            feedback: { value: 'valid' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregateStringAssessments(traces, 'category');
      expect(result.totalCount).toBe(1);
      expect(result.valueCounts.get('valid')).toBe(1);
      expect(result.valueCounts.get('invalid')).toBeUndefined();
    });

    it('skips session-level assessments', () => {
      const traces = [
        createTrace([
          {
            assessment_name: 'category',
            feedback: { value: 'session-value' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
            metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'true' },
          },
          {
            assessment_name: 'category',
            feedback: { value: 'trace-value' },
            source: { source_type: 'LLM_JUDGE', source_id: 'test' },
          },
        ]),
      ];

      const result = aggregateStringAssessments(traces, 'category');
      expect(result.totalCount).toBe(1);
      expect(result.valueCounts.get('trace-value')).toBe(1);
      expect(result.valueCounts.get('session-value')).toBeUndefined();
    });
  });
});
