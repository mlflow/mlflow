import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

// Written by the MLflow SDK on feedback from a registered scorer (AssessmentMetadataKey in
// mlflow/tracing/constant.py); keep these keys in sync with it.
const SCORER_NAME_METADATA_KEY = 'mlflow.assessment.scorerName';
const SCORER_VERSION_METADATA_KEY = 'mlflow.assessment.scorerVersion';

export interface RegisteredScorerReference {
  scorerName: string;
  scorerVersion?: number;
}

const parseScorerVersion = (value: string | undefined): number | undefined => {
  if (value === undefined || !/^[1-9]\d*$/.test(value)) {
    return undefined;
  }
  const version = Number(value);
  return Number.isSafeInteger(version) ? version : undefined;
};

export const parseRegisteredScorerReference = (
  metadata: Record<string, string> | undefined,
): RegisteredScorerReference | undefined => {
  const scorerName = metadata?.[SCORER_NAME_METADATA_KEY];
  if (typeof scorerName !== 'string' || !scorerName) {
    return undefined;
  }
  return { scorerName, scorerVersion: parseScorerVersion(metadata?.[SCORER_VERSION_METADATA_KEY]) };
};

/**
 * The registered scorer that produced every assessment named `assessmentName` on these traces. Undefined when
 * any of them has no registered scorer or they disagree, so a link never points at the wrong scorer.
 */
export const getConsistentScorerReference = (
  traces: ModelTraceInfoV3[],
  assessmentName: string,
): RegisteredScorerReference | undefined => {
  const references = traces.flatMap((trace) =>
    (trace.assessments ?? [])
      .filter((assessment) => assessment.assessment_name === assessmentName)
      .map((assessment) => parseRegisteredScorerReference(assessment.metadata)),
  );
  const [firstReference] = references;
  if (!firstReference) {
    return undefined;
  }
  return references.every(
    (reference) =>
      reference?.scorerName === firstReference.scorerName && reference.scorerVersion === firstReference.scorerVersion,
  )
    ? firstReference
    : undefined;
};
