import { difference, groupBy, isNil, isNumber, isPlainObject, orderBy, zipObject } from 'lodash';

import { KnownEvaluationResultAssessmentMetadataFields, KnownEvaluationResultAssessmentName } from '../enum';
import type {
  EvaluationArtifactTableEntryAssessment,
  EvaluationArtifactTableEntryEvaluation,
  EvaluationArtifactTableEntryMetric,
  RawGenaiEvaluationArtifactResponse,
  RunEvaluationResultAssessment,
  RunEvaluationResultMetric,
  RunEvaluationTracesDataEntry,
  RunEvaluationTracesRetrievalChunk,
} from '../types';

export const isEvaluationResultOverallAssessment = (assessmentEntry: RunEvaluationResultAssessment) =>
  assessmentEntry.metadata?.[KnownEvaluationResultAssessmentMetadataFields.IS_OVERALL_ASSESSMENT] === true;

export const isEvaluationResultPerRetrievalChunkAssessment = (assessmentEntry: RunEvaluationResultAssessment) =>
  isNumber(assessmentEntry.metadata?.[KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]);

/**
 * Checks if the given value is a retrieved context.
 * A retrieved context is a list of objects with a `doc_uri` and `content` field.
 */
export const isRetrievedContext = (value: any): boolean => {
  return Array.isArray(value) && value.every((v) => isPlainObject(v) && 'doc_uri' in v && 'content' in v);
};

/**
 * Extracts the first retrieved context value from the given record.
 * Returns undefined if no retrieved context is found.
 */
const getFirstRetrievedContextValue = (
  record?: Record<string, any>,
): { doc_uri: string; content: string }[] | undefined => {
  return Object.values(record || {}).find(isRetrievedContext) as { doc_uri: string; content: string }[];
};

/**
 * Extracts the retrieval chunks from the given outputs, targets and per-chunk assessments.
 */
export const extractRetrievalChunks = (
  outputs?: Record<string, any>,
  targets?: Record<string, any>,
  perChunkAssessments?: RunEvaluationResultAssessment[],
): RunEvaluationTracesRetrievalChunk[] => {
  // Only support one retrieved context for now, first one is used
  const retrievedContext = getFirstRetrievedContextValue(outputs) || [];
  const expectedRetrievedContext = getFirstRetrievedContextValue(targets);

  return retrievedContext.map((retrievedContext, index) => {
    const target = expectedRetrievedContext?.[index];
    const assessments = (perChunkAssessments || []).filter(
      (assessment) => (assessment.metadata || {})[KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX] === index,
    );
    // Group detailed assessments by name
    const retrievalAssessmentsByName = groupBy(assessments, 'name');
    // Ensure each group is sorted by timestamp in descending order
    Object.keys(retrievalAssessmentsByName).forEach((key) => {
      retrievalAssessmentsByName[key] = orderBy(retrievalAssessmentsByName[key], 'timestamp', 'desc');
    });

    return {
      docUrl: retrievedContext.doc_uri,
      content: retrievedContext.content,
      retrievalAssessmentsByName: retrievalAssessmentsByName,
      target: target?.content,
    };
  });
};

export function parseRawTableArtifact<T>(artifactData?: RawGenaiEvaluationArtifactResponse): T | undefined {
  if (!artifactData) {
    return undefined;
  }
  const { columns, data, filename } = artifactData;

  if (!columns || !Array.isArray(columns)) {
    throw new SyntaxError(`Artifact ${filename} is malformed, it does not contain "columns" array`);
  }
  if (!data || !Array.isArray(data)) {
    throw new SyntaxError(`Artifact ${filename} is malformed, it does not contain "data" array`);
  }

  const normalizedColumns = columns.map((column, index) => column ?? `column_${index}`);

  return data.map((row) => zipObject(normalizedColumns, row)) as T;
}

export function mergeMetricsAndAssessmentsWithEvaluations(
  evaluations: EvaluationArtifactTableEntryEvaluation[],
  metrics?: EvaluationArtifactTableEntryMetric[],
  assessments?: EvaluationArtifactTableEntryAssessment[],
): RunEvaluationTracesDataEntry[] {
  // Group metrics by evaluation_id.
  const metricsByEvaluation = (metrics || []).reduce<Record<string, Record<string, RunEvaluationResultMetric>>>(
    (acc, entry: any) => {
      if (!acc[entry.evaluation_id]) {
        acc[entry.evaluation_id] = {};
      }
      const { key, value, timestamp } = entry;
      acc[entry.evaluation_id][key] = { key, value, timestamp };
      return acc;
    },
    {},
  );

  // Group assessments by evaluation_id.
  const assessmentsByEvaluation = (assessments || []).reduce<Record<string, RunEvaluationResultAssessment[]>>(
    (acc, entry: EvaluationArtifactTableEntryAssessment) => {
      if (!acc[entry.evaluation_id]) {
        acc[entry.evaluation_id] = [];
      }
      acc[entry.evaluation_id].push({
        booleanValue: !isNil(entry.boolean_value) ? Boolean(entry.boolean_value) : entry.boolean_value,
        numericValue: !isNil(entry.numeric_value) ? Number(entry.numeric_value) : entry.numeric_value,
        stringValue: !isNil(entry.string_value) ? String(entry.string_value) : entry.string_value,
        metadata: entry.metadata || {},
        ...(entry.error_code && { errorCode: entry.error_code }),
        ...(entry.error_message && { errorMessage: entry.error_message }),
        name: entry.name,
        rationale: entry.rationale || null,
        source: {
          metadata: entry.source?.metadata ?? {},
          sourceId: entry.source?.source_id,
          sourceType: entry.source?.source_type,
        },
        timestamp: entry.timestamp,
      });
      return acc;
    },
    {},
  );

  return evaluations.map((entry: any) => {
    // Get all assessments for the evaluation and group them by name
    const allAssessmentsSorted = orderBy(assessmentsByEvaluation[entry.evaluation_id] || [], 'timestamp', 'desc');
    const overallAssessments: RunEvaluationResultAssessment[] = allAssessmentsSorted
      .filter(isEvaluationResultOverallAssessment)
      .map((assessment) => {
        // Find the "[assessment_name]" prefix and convert it to the rootCauseAssessment, removing it from the prefix.
        // The format is: "[assessment_name] rationale **Suggested Actions**: suggestedActions"
        const match = assessment.rationale?.match(/^\[(.*?)\](.*?)(?:\*\*Suggested Actions\*\*:(.*))?$/s);

        const assessmentName = match ? match[1]?.trim() : undefined;
        const newRationale = match ? match[2]?.trim() : undefined;
        const suggestedActions = match ? match[3]?.trim() : undefined;

        assessment.rationale = newRationale || assessment.rationale;
        const result: RunEvaluationResultAssessment = {
          ...assessment,
          rootCauseAssessment: !isNil(assessmentName) ? { assessmentName, suggestedActions } : undefined,
        };
        return result;
      });
    if (overallAssessments.length === 0) {
      // In the special case where there is no overall assessment, we create a null here so the UI can render it.
      overallAssessments.push({
        name: KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
        rationale: null,
        source: {
          sourceType: 'AI_JUDGE',
          sourceId: 'UNKNOWN',
          metadata: {},
        },
        metadata: {},
        numericValue: null,
        booleanValue: null,
        stringValue: null,
        timestamp: null,
      });
    }
    // TODO(nsthorat): perRetrievalChunkAsessments should be treated differently than other methods, and removed from the overall metrics.
    const perRetrievalChunkAssessments = allAssessmentsSorted.filter(isEvaluationResultPerRetrievalChunkAssessment);

    // All assessments that are not overall or per retrieval chunk are response assessments
    const responseAssessments = difference(allAssessmentsSorted, overallAssessments, perRetrievalChunkAssessments);

    // Group response assessments by name
    const responseAssessmentsByName = groupBy(responseAssessments, 'name');
    // Ensure each group is sorted by timestamp in descending order
    Object.keys(responseAssessmentsByName).forEach((key) => {
      responseAssessmentsByName[key] = orderBy(responseAssessmentsByName[key], 'timestamp', 'desc');
    });

    return {
      evaluationId: entry.evaluation_id,
      requestId: entry.request_id,
      inputs: entry.inputs,
      inputsId: entry.inputs_id,
      outputs: entry.outputs ?? {},
      targets: entry.targets ?? {},
      ...(entry.error_code && { errorCode: entry.error_code }),
      ...(entry.error_message && { errorMessage: entry.error_message }),
      overallAssessments,
      responseAssessmentsByName,
      metrics: metricsByEvaluation[entry.evaluation_id] || {},
      retrievalChunks: extractRetrievalChunks(entry.outputs, entry.targets, perRetrievalChunkAssessments),
    };
  });
}
