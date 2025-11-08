import { useCallback, useState } from 'react';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '../../../../redux-types';
import { GET_EVALUATION_TABLE_ARTIFACT, uploadArtifactApi } from '../../../actions';
import type {
  RunEvaluationResultAssessment,
  RunEvaluationResultAssessmentDraft,
} from '@databricks/web-shared/genai-traces-table';
import type { RawEvaluationArtifact } from '../../../sdk/EvaluationArtifactService';
import { parseEvaluationTableArtifact } from '../../../sdk/EvaluationArtifactService';
import { getArtifactChunkedText, getArtifactLocationUrl } from '../../../../common/utils/ArtifactUtils';
import { isArray, isEqual } from 'lodash';
import Utils from '../../../../common/utils/Utils';
import { fulfilled } from '../../../../common/utils/ActionUtils';
import { ASSESSMENTS_ARTIFACT_FILE_NAME } from '../constants';

/**
 * Local utility function to fetch existing raw assessments artifact data.
 */
const fetchExistingRawAssessmentsArtifactData = async (runUuid: string): Promise<RawEvaluationArtifact> => {
  const fullArtifactSrcPath = getArtifactLocationUrl(ASSESSMENTS_ARTIFACT_FILE_NAME, runUuid);

  const fileContents = await getArtifactChunkedText(fullArtifactSrcPath).then((artifactContent) =>
    JSON.parse(artifactContent),
  );

  if (!isArray(fileContents.data) || !isArray(fileContents.columns)) {
    throw new Error('Artifact is malformed and/or not valid JSON');
  }

  return fileContents;
};

// We have to keep the list of fields in sync with the schema defined in MLflow.
// See mlflow/evaluation/utils.py#_get_assessments_dataframe_schema for the schema definition.
const assessmentsToEvaluationArtifactJSONRows = (
  evaluationId: string,
  assessments: RunEvaluationResultAssessment[],
): RawEvaluationArtifact['data'] => {
  return assessments.map((assessment) => {
    return [
      evaluationId,
      assessment.name,
      {
        source_type: assessment.source?.sourceType,
        source_id: assessment.source?.sourceId,
        source_metadata: assessment.source?.metadata,
      },
      assessment.timestamp || null,
      assessment.booleanValue || null,
      assessment.numericValue || null,
      assessment.stringValue || null,
      assessment.rationale || null,
      assessment.metadata || null,
      null, // error_code
      null, // error_message
    ];
  });
};

/**
 * Iterates through the existing assessments and removes the ones with matching sources from the pending assessments.
 * Accepts the existing assessments artifact and the pending assessments to be saved.
 */
const filterExistingAssessmentsBySource = (
  evaluationId: string,
  existingAssessmentsArtifact: RawEvaluationArtifact,
  pendingAssessments: RunEvaluationResultAssessmentDraft[],
) => {
  // Parse the existing assessments artifact file
  const existingAssessmentsFile = parseEvaluationTableArtifact(
    ASSESSMENTS_ARTIFACT_FILE_NAME,
    existingAssessmentsArtifact,
  );

  // Map the sources of the pending assessments to the format of the existing assessments
  const sourcesInPendingAssessments = pendingAssessments.map(({ name, source }) => ({
    name,
    source: source
      ? {
          source_type: source.sourceType,
          source_id: source.sourceId,
          source_metadata: source.metadata,
        }
      : undefined,
  }));

  // Find the entries in the existing assessments that have the same evaluation_id and source as the pending assessments
  const entriesToBeRemoved = existingAssessmentsFile.entries.filter(
    ({ evaluation_id, name, source }) =>
      evaluationId === evaluation_id &&
      sourcesInPendingAssessments.find((incomingEntry) => isEqual({ name, source }, incomingEntry)),
  );

  // Find the indexes of the entries to be removed
  const indexesToBeRemoved = entriesToBeRemoved.map((entry) => existingAssessmentsFile.entries.indexOf(entry));

  // Remove the entries from the existing assessments
  return existingAssessmentsArtifact.data.filter((_, index) => !indexesToBeRemoved.includes(index));
};

export const useSavePendingEvaluationAssessments = () => {
  const dispatch = useDispatch<ThunkDispatch>();

  const [isSaving, setIsSaving] = useState(false);

  const savePendingAssessments = useCallback(
    async (runUuid: string, evaluationId: string, pendingAssessmentEntries: RunEvaluationResultAssessmentDraft[]) => {
      try {
        setIsSaving(true);
        // Start with fetching existing state of the data so we have the fresh one
        const existingAssessmentsArtifact = await fetchExistingRawAssessmentsArtifactData(runUuid);

        // Map the assessments to the JSON file format
        const newData = assessmentsToEvaluationArtifactJSONRows(evaluationId, pendingAssessmentEntries);

        // Filter out the existing assessments that have the same source as the pending assessments
        const filteredExistingData = filterExistingAssessmentsBySource(
          evaluationId,
          existingAssessmentsArtifact,
          pendingAssessmentEntries,
        );

        // Upload the new artifact file. Explicitly "await" for the result so we can catch any errors.
        await dispatch(
          uploadArtifactApi(runUuid, ASSESSMENTS_ARTIFACT_FILE_NAME, {
            columns: existingAssessmentsArtifact.columns,
            data: [...filteredExistingData, ...newData],
          }),
        );

        dispatch({
          type: fulfilled(GET_EVALUATION_TABLE_ARTIFACT),
          payload: parseEvaluationTableArtifact(ASSESSMENTS_ARTIFACT_FILE_NAME, {
            columns: existingAssessmentsArtifact.columns,
            data: [...filteredExistingData, ...newData],
          }),
          meta: { runUuid, artifactPath: ASSESSMENTS_ARTIFACT_FILE_NAME },
        });
      } catch (e: any) {
        Utils.logErrorAndNotifyUser(e.message || e);
        throw e;
      } finally {
        setIsSaving(false);
      }
    },
    [dispatch],
  );

  return { savePendingAssessments, isSaving };
};
