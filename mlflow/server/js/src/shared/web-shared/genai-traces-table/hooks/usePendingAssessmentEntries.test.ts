import { act, renderHook } from '@testing-library/react';

import { usePendingAssessmentEntries } from './usePendingAssessmentEntries';
import {
  KnownEvaluationResultAssessmentMetadataFields,
  KnownEvaluationResultAssessmentName,
} from '../components/GenAiEvaluationTracesReview.utils';
import type { RunEvaluationResultAssessmentDraft } from '../types';

describe('usePendingAssessmentEntries', () => {
  let testEmptyEvaluationResult: any;
  beforeEach(() => {
    testEmptyEvaluationResult = {
      overallAssessments: [],
      responseAssessmentsByName: {},
    };
  });
  it('should add new overall assessment to empty evaluation set', () => {
    const assessment: RunEvaluationResultAssessmentDraft = {
      name: 'overall_assessment',
      value: 'some value',
      metadata: {
        [KnownEvaluationResultAssessmentMetadataFields.IS_OVERALL_ASSESSMENT]: true,
      },
    } as any;

    const { result } = renderHook(() => usePendingAssessmentEntries(testEmptyEvaluationResult));

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(0);
    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({});

    act(() => {
      result.current.upsertAssessment(assessment);
    });

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(1);
    expect(result.current.draftEvaluationResult.overallAssessments[0]).toEqual(assessment);
    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({});
  });

  it('should update existing overall assessment', () => {
    const existingAssessment: RunEvaluationResultAssessmentDraft = {
      name: 'overall_assessment',
      value: 'some value',
      metadata: {
        [KnownEvaluationResultAssessmentMetadataFields.IS_OVERALL_ASSESSMENT]: true,
      },
    } as any;

    const newAssessment = {
      ...existingAssessment,
      value: 'new value',
    };

    const { result } = renderHook(() =>
      usePendingAssessmentEntries({
        ...testEmptyEvaluationResult,
        overallAssessments: [existingAssessment],
      }),
    );

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(1);

    act(() => {
      result.current.upsertAssessment(newAssessment);
    });

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(2);
    expect(result.current.draftEvaluationResult.overallAssessments).toEqual([newAssessment, existingAssessment]);

    const anotherAssessment = {
      ...existingAssessment,
      value: 'another value',
    };

    act(() => {
      result.current.upsertAssessment(anotherAssessment);
    });

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(2);
    expect(result.current.draftEvaluationResult.overallAssessments).toEqual([anotherAssessment, existingAssessment]);
  });

  it('should add new detailed assessment to empty evaluation set', () => {
    const assessment: RunEvaluationResultAssessmentDraft = {
      name: KnownEvaluationResultAssessmentName.GROUNDEDNESS,
      value: 'yes',
    } as any;

    const { result } = renderHook(() => usePendingAssessmentEntries(testEmptyEvaluationResult));

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(0);
    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({});

    act(() => {
      result.current.upsertAssessment(assessment);
    });

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(0);
    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({
      [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: [assessment],
    });
  });

  it('should update existing detailed assessment', () => {
    const newAssessment: RunEvaluationResultAssessmentDraft = {
      name: KnownEvaluationResultAssessmentName.GROUNDEDNESS,
      value: 'no',
    } as any;

    const existingAssessment: RunEvaluationResultAssessmentDraft = {
      name: KnownEvaluationResultAssessmentName.GROUNDEDNESS,
      value: 'yes',
    } as any;

    const { result } = renderHook(() =>
      usePendingAssessmentEntries({
        ...testEmptyEvaluationResult,
        responseAssessmentsByName: {
          [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: [existingAssessment],
        },
      }),
    );

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(0);
    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({
      [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: [existingAssessment],
    });

    act(() => {
      result.current.upsertAssessment(newAssessment);
    });

    expect(result.current.draftEvaluationResult.overallAssessments).toHaveLength(0);
    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({
      [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: [newAssessment, existingAssessment],
    });

    const anotherAssessment = {
      ...existingAssessment,
      value: 'weak no',
    };

    act(() => {
      result.current.upsertAssessment(anotherAssessment);
    });

    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({
      [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: [anotherAssessment, existingAssessment],
    });
  });

  it('should update existing detailed retrieval chunk assessment', () => {
    const newAssessment: RunEvaluationResultAssessmentDraft = {
      name: KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE,
      value: 'no',
      metadata: {
        [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 1,
      },
    } as any;

    const existingAssessments: RunEvaluationResultAssessmentDraft[] = [
      {
        name: KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE,
        value: 'yes',
        metadata: {
          [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 0,
        },
      },
      {
        name: KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE,
        value: 'yes',
        metadata: {
          [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 1,
        },
      },
    ] as any;

    const { result } = renderHook(() =>
      usePendingAssessmentEntries({
        ...testEmptyEvaluationResult,
        retrievalChunks: [
          {
            content: 'abc',
            retrievalAssessmentsByName: {
              [KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE]: [existingAssessments[0]],
            },
          },
          {
            content: 'xyz',
            retrievalAssessmentsByName: {
              [KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE]: [existingAssessments[1]],
            },
          },
        ],
      }),
    );

    act(() => {
      result.current.upsertAssessment(newAssessment);
    });

    // Assessments in retrieval chunk #0 should be unchanged
    expect(
      result.current.draftEvaluationResult.retrievalChunks?.[0].retrievalAssessmentsByName[
        KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE
      ],
    ).toEqual([existingAssessments[0]]);

    // Assessments in retrieval chunk #1 should be updated with a new entry
    expect(
      result.current.draftEvaluationResult.retrievalChunks?.[1].retrievalAssessmentsByName[
        KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE
      ],
    ).toEqual([newAssessment, existingAssessments[1]]);

    const assessmentWithAnotherChange = { ...newAssessment, stringValue: 'weak yes' };

    act(() => {
      result.current.upsertAssessment(assessmentWithAnotherChange);
    });

    // Pending assessment in retrieval chunk #1 should be replaced with a new one
    expect(
      result.current.draftEvaluationResult.retrievalChunks?.[1].retrievalAssessmentsByName[
        KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE
      ],
    ).toEqual([assessmentWithAnotherChange, existingAssessments[1]]);
  });

  it('should add detailed retrieval chunk assessment in correct group and order', () => {
    const newAssessment: RunEvaluationResultAssessmentDraft = {
      name: 'a1',
      value: 'no',
      metadata: {
        [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 0,
      },
      timestamp: 3500,
    } as any;

    const existingAssessments: RunEvaluationResultAssessmentDraft[] = [
      {
        name: 'a1',
        value: 'yes',
        metadata: {
          [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 0,
        },
        timestamp: 4000,
      },
      {
        name: 'a1',
        value: 'weak yes',
        metadata: {
          [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 0,
        },
        timestamp: 3000,
      },
      {
        name: 'a2',
        value: 'yes',
        metadata: {
          [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 0,
        },
        timestamp: 2000,
      },
    ] as any;

    const { result } = renderHook(() =>
      usePendingAssessmentEntries({
        ...testEmptyEvaluationResult,
        retrievalChunks: [
          {
            content: 'abc',
            retrievalAssessmentsByName: {
              a1: [existingAssessments[0], existingAssessments[1]],
              a2: [existingAssessments[2]],
            },
          },
        ],
      }),
    );

    act(() => {
      result.current.upsertAssessment(newAssessment);
    });

    // Assessments in retrieval chunk should be updated with a new entry in the correct group and order
    expect(result.current.draftEvaluationResult.retrievalChunks?.[0].retrievalAssessmentsByName).toEqual({
      a1: [existingAssessments[0], newAssessment, existingAssessments[1]],
      a2: [existingAssessments[2]],
    });
  });

  it('should add new retrieval chunk assessment', () => {
    const newAssessment: RunEvaluationResultAssessmentDraft = {
      name: KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE,
      value: 'yes',
      metadata: {
        [KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]: 0,
      },
    } as any;

    const { result } = renderHook(() =>
      usePendingAssessmentEntries({
        ...testEmptyEvaluationResult,
        retrievalChunks: [
          {
            content: 'abc',
            docUrl: 'https://mlflow.org/docs/abc',
          },
        ],
      }),
    );

    act(() => {
      result.current.upsertAssessment(newAssessment);
    });

    expect(
      result.current.draftEvaluationResult.retrievalChunks?.[0].retrievalAssessmentsByName[
        KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE
      ],
    ).toEqual([newAssessment]);
  });

  it('should support non-mapped detailed assessments', () => {
    const { result } = renderHook(() => usePendingAssessmentEntries(testEmptyEvaluationResult));

    const alphaAssessment: RunEvaluationResultAssessmentDraft = {
      name: 'test-alpha',
      value: 'alpha',
    } as any;
    const betaAssessment: RunEvaluationResultAssessmentDraft = {
      name: 'test-beta',
      value: 'beta',
    } as any;

    act(() => {
      result.current.upsertAssessment(alphaAssessment);
      result.current.upsertAssessment(betaAssessment);
    });

    expect(result.current.draftEvaluationResult.responseAssessmentsByName).toEqual({
      'test-alpha': [
        {
          name: 'test-alpha',
          value: 'alpha',
        },
      ],
      'test-beta': [
        {
          name: 'test-beta',
          value: 'beta',
        },
      ],
    });
  });
});
