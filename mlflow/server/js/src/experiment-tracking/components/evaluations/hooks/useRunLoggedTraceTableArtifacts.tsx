import { intersection } from 'lodash';
import type { KeyValueEntity } from '../../../../common/types';
import { extractLoggedTablesFromRunTags } from '../../../utils/ArtifactUtils';
import { GenAiTraceEvaluationArtifactFile } from '@databricks/web-shared/genai-traces-table';
import { useMemo } from 'react';

/**
 * Returns the list of known evaluation table artifacts that are logged for a run based on its tags.
 */
export const useRunLoggedTraceTableArtifacts = (runTags?: Record<string, KeyValueEntity>) =>
  useMemo(() => {
    if (!runTags) {
      return [];
    }
    return intersection(extractLoggedTablesFromRunTags(runTags), [
      GenAiTraceEvaluationArtifactFile.Evaluations,
      GenAiTraceEvaluationArtifactFile.Metrics,
      GenAiTraceEvaluationArtifactFile.Assessments,
    ]) as GenAiTraceEvaluationArtifactFile[];
  }, [runTags]);
