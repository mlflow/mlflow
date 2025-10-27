import { flatMap, groupBy, orderBy, uniq } from 'lodash';
import { useCallback, useMemo, useReducer } from 'react';

import {
  KnownEvaluationResultAssessmentMetadataFields,
  getEvaluationResultAssessmentChunkIndex,
  getEvaluationResultAssessmentValue,
  isEvaluationResultOverallAssessment,
  isEvaluationResultPerRetrievalChunkAssessment,
} from '../components/GenAiEvaluationTracesReview.utils';
import type { RunEvaluationResultAssessmentDraft, RunEvaluationTracesDataEntry } from '../types';

export const usePendingAssessmentEntries = (evaluationResult: RunEvaluationTracesDataEntry) => {
  const [pendingAssessments, dispatch] = useReducer(
    (
      state: RunEvaluationResultAssessmentDraft[],
      action:
        | {
            type: 'upsertAssessment';
            payload: RunEvaluationResultAssessmentDraft;
          }
        | {
            type: 'resetPendingAssessments';
          },
    ): RunEvaluationResultAssessmentDraft[] => {
      switch (action.type) {
        case 'resetPendingAssessments':
          return [];
        case 'upsertAssessment':
          const existingPendingAssessment = state.find(
            (assessment) =>
              assessment.name === action.payload.name &&
              getEvaluationResultAssessmentChunkIndex(assessment) ===
                getEvaluationResultAssessmentChunkIndex(action.payload),
          );

          // If the incoming assessment is already in the pending list, update it.
          if (existingPendingAssessment) {
            return state.map((assessment) => {
              if (assessment === existingPendingAssessment) {
                // Special case: handling existing assessment without value. The value of the incoming assessment
                // will be used to update the name of the assessment.
                if (getEvaluationResultAssessmentValue(assessment) === '' && action.payload.stringValue) {
                  return { ...action.payload, stringValue: '', name: action.payload.stringValue };
                }
                return action.payload;
              }
              return assessment;
            });
          }
          return [action.payload, ...state];
      }
    },
    [],
  );

  const overallAssessmentList = useMemo(() => {
    if (!evaluationResult) {
      return [];
    }
    const pendingOverallAssessment = pendingAssessments.find(isEvaluationResultOverallAssessment);
    if (pendingOverallAssessment) {
      return [pendingOverallAssessment, ...evaluationResult.overallAssessments];
    }
    return evaluationResult.overallAssessments;
  }, [evaluationResult, pendingAssessments]);

  const responseAssessmentMap = useMemo(() => {
    if (!evaluationResult) {
      return {};
    }

    const allAssessmentNames = uniq([
      ...Object.keys(evaluationResult.responseAssessmentsByName),
      ...pendingAssessments
        .filter(
          (assessment) =>
            !isEvaluationResultOverallAssessment(assessment) &&
            !isEvaluationResultPerRetrievalChunkAssessment(assessment),
        )
        .map((assessment) => assessment.name),
    ]);

    return Object.fromEntries(
      allAssessmentNames.map((key) => {
        const pendingAssessmentForType = pendingAssessments.filter((assessment) => assessment.name === key);
        return [key, [...pendingAssessmentForType, ...(evaluationResult.responseAssessmentsByName[key] || [])]];
      }),
    );
  }, [evaluationResult, pendingAssessments]);

  const retrievalChunksWithDraftAssessments = useMemo(() => {
    const perChunkAssessments = pendingAssessments.filter(isEvaluationResultPerRetrievalChunkAssessment);
    return evaluationResult.retrievalChunks?.map((chunk, index) => {
      const pendingAssessmentForChunk = perChunkAssessments.filter(
        (assessment) => assessment.metadata?.[KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX] === index,
      );
      const existingAssessments = flatMap(chunk.retrievalAssessmentsByName ?? {});
      // Group detailed assessments by name
      const retrievalAssessmentsByName = groupBy([...pendingAssessmentForChunk, ...existingAssessments], 'name');
      // Ensure each group is sorted by timestamp in descending order
      Object.keys(retrievalAssessmentsByName).forEach((key) => {
        retrievalAssessmentsByName[key] = orderBy(retrievalAssessmentsByName[key], 'timestamp', 'desc');
      });
      return {
        ...chunk,
        retrievalAssessmentsByName: retrievalAssessmentsByName,
      };
    });
  }, [pendingAssessments, evaluationResult]);

  const draftEvaluationResult = useMemo(() => {
    return {
      ...evaluationResult,
      overallAssessments: overallAssessmentList,
      responseAssessmentsByName: responseAssessmentMap,
      retrievalChunks: retrievalChunksWithDraftAssessments,
    };
  }, [evaluationResult, overallAssessmentList, responseAssessmentMap, retrievalChunksWithDraftAssessments]);

  const upsertAssessment = useCallback(
    (assessment: RunEvaluationResultAssessmentDraft) => dispatch({ type: 'upsertAssessment', payload: assessment }),
    [],
  );

  const resetPendingAssessments = useCallback(() => dispatch({ type: 'resetPendingAssessments' }), []);

  return {
    pendingAssessments,
    draftEvaluationResult,
    upsertAssessment,
    resetPendingAssessments,
  };
};
