import type { AssessmentInfo, EvalTraceComparisonEntry } from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { getAssessmentAggregates } from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

/** Scorer aggregates for the eval run comparison, where an errored score is not a real value. */
export const getEvalRunComparisonAssessmentAggregates = (
  assessmentInfo: AssessmentInfo,
  evalResults: EvalTraceComparisonEntry[],
) => getAssessmentAggregates(assessmentInfo, evalResults, [], { excludeErroredValues: shouldEnableNewEvalRunsUi() });
