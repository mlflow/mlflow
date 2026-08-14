export enum GenAiTraceEvaluationArtifactFile {
  Evaluations = '_evaluations.json',
  Assessments = '_assessments.json',
  Metrics = '_metrics.json',
}

export enum KnownEvaluationResultAssessmentMetadataFields {
  IS_OVERALL_ASSESSMENT = 'is_overall_assessment',
  CHUNK_INDEX = 'chunk_index',
  IS_COPIED_FROM_AI = 'is_copied_from_ai',
}

export enum KnownEvaluationResultAssessmentName {
  OVERALL_ASSESSMENT = 'overall_assessment',
  SAFETY = 'safety',
  GROUNDEDNESS = 'groundedness',
  CORRECTNESS = 'correctness',
  RELEVANCE_TO_QUERY = 'relevance_to_query',
  CHUNK_RELEVANCE = 'chunk_relevance',
  CONTEXT_SUFFICIENCY = 'context_sufficiency',
  GUIDELINE_ADHERENCE = 'guideline_adherence',
  GLOBAL_GUIDELINE_ADHERENCE = 'global_guideline_adherence',
  TOTAL_TOKEN_COUNT = 'agent/total_token_count',
  TOTAL_INPUT_TOKEN_COUNT = 'agent/total_input_token_count',
  TOTAL_OUTPUT_TOKEN_COUNT = 'agent/total_output_token_count',
  DOCUMENT_RECALL = 'retrieval/ground_truth/document_recall',
  DOCUMENT_RATINGS = 'retrieval/ground_truth/document_ratings',
}
