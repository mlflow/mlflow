import { useCallback, useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { isEvaluatingSessionsInScorersEnabled, isRunningScorersEnabled } from '../../../common/utils/FeatureUtils';
import { fetchOrFail, getAjaxUrl } from '../../../common/utils/FetchUtils';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import type {
  ModelTraceLocationMlflowExperiment,
  ModelTraceLocationUcSchema,
} from '@databricks/web-shared/model-trace-explorer';
import {
  extractFromTrace,
  buildSystemPrompt,
  buildUserPrompt,
  extractTemplateVariables,
} from '../../utils/evaluationUtils';
import { searchMlflowTracesQueryFn, SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import { DEFAULT_TRACE_COUNT, RETRIEVAL_ASSESSMENTS, ScorerEvaluationScope } from './constants';
import {
  extractInputs,
  extractOutputs,
  extractRetrievalContext,
  extractExpectations,
  type TraceRetrievalContexts,
  type RetrievalContext,
} from '../../utils/TraceUtils';
import { EvaluateChatCompletionsParams, EvaluateTracesParams } from './types';
import { useGetTraceIdsForEvaluation } from './useGetTracesForEvaluation';
import { getMlflowTraceV3ForEvaluation, JudgeEvaluationResult } from './useEvaluateTraces.common';

/**
 * Response from the chat completions API
 */
interface ChatCompletionsResponse {
  output: string | null;
  error_code: string | null;
  error_message: string | null;
}

/**
 * Request to the chat completions API
 */
interface ChatCompletionsRequest {
  user_prompt: string;
  system_prompt?: string | null;
  experiment_id: string;
}

async function callChatCompletions(
  userPrompt: string,
  systemPrompt: string | null,
  experimentId: string,
): Promise<ChatCompletionsResponse> {
  const requestBody: ChatCompletionsRequest = {
    user_prompt: userPrompt,
    ...(systemPrompt && { system_prompt: systemPrompt }),
    experiment_id: experimentId,
  };

  const response = await fetchOrFail(getAjaxUrl('ajax-api/2.0/agents/chat-completions'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  return response.json();
}

/**
 * Parses the judge response to extract result and rationale
 */
function parseJudgeResponse(output: string | null): {
  result: string | null;
  rationale: string | null;
} {
  if (!output) {
    return { result: null, rationale: null };
  }

  try {
    // Try to parse as JSON
    const parsed = JSON.parse(output);
    return {
      result: parsed.result || null,
      rationale: parsed.rationale || null,
    };
  } catch {
    // If not valid JSON, return the output as the result
    return {
      result: output,
      rationale: null,
    };
  }
}

/**
 * Type guard to check if params are for chat completions
 */
function isChatCompletionsParams(params: EvaluateTracesParams): params is EvaluateChatCompletionsParams {
  return 'judgeInstructions' in params;
}

/**
 * Response from the chat assessments API
 */
interface ChatAssessmentsResponse {
  result: {
    response_assessment: {
      ratings: Record<
        string,
        {
          value: string | any; // Can be a string or structured object with categorical_value, double_value, etc.
          rationale: string;
          error?: string;
        }
      >;
    };
    retrieval_assessment?: any;
  };
}

/**
 * Request to the chat assessments API
 */
interface ChatAssessmentsRequest {
  assessment_input: {
    chat_request?: string;
    chat_response?: string;
    retrieval_context?: {
      retrieved_documents: Array<{
        doc_uri: string;
        content: string;
      }>;
    };
    ground_truth?: {
      expected_chat_response: string;
    };
    guidelines?: string[];
    guidelines_context?: {
      request: string;
      response?: string;
    };
  };
  requested_assessments: Array<{
    assessment_name: string;
    assessment_examples?: any[];
  }>;
  experiment_id: string;
}

/**
 * Call the chat assessments API
 */
async function callChatAssessments(
  chatRequest: string,
  chatResponse: string | null,
  retrievalSpan: RetrievalContext | null,
  groundTruth: Record<string, any>,
  guidelines: string[] | null,
  requestedAssessments: Array<{ assessment_name: string; assessment_examples?: any[] }>,
  experimentId: string,
  isGuidelinesAssessment: boolean = false,
): Promise<ChatAssessmentsResponse> {
  // Use retrieval span documents directly without transformation
  // RetrievalContext already has { documents: [...] } format
  const retrievalContext = retrievalSpan ? { retrieved_documents: retrievalSpan.documents } : null;

  // For guidelines assessment, include both chat_request (required) AND guidelines_context
  const assessmentInput: ChatAssessmentsRequest['assessment_input'] = {
    chat_request: chatRequest,
    ...(chatResponse && { chat_response: chatResponse }),
    ...(retrievalContext && { retrieval_context: retrievalContext }),
    ...(isGuidelinesAssessment && {
      guidelines_context: {
        request: chatRequest,
        ...(chatResponse && { response: chatResponse }),
      },
    }),
    ...(!isGuidelinesAssessment &&
      Object.keys(groundTruth).length > 0 && {
        ground_truth: {
          expected_chat_response: JSON.stringify(groundTruth),
        },
      }),
    ...(guidelines && guidelines.length > 0 && { guidelines }),
  };

  const requestBody: ChatAssessmentsRequest = {
    assessment_input: assessmentInput,
    requested_assessments: requestedAssessments,
    experiment_id: experimentId,
  };

  const response = await fetchOrFail(getAjaxUrl('ajax-api/2.0/agents/chat-assessments'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  return response.json();
}

/**
 * Parse assessment response to extract result and rationale
 * Assumes single assessment in the response
 *
 * Handles two types of assessments:
 * 1. Response assessments (correctness, relevance_to_query, harmfulness, guidelines)
 *    - Located in: response.result.response_assessment.ratings
 * 2. Retrieval assessments (chunk_relevance, groundedness, context_sufficiency)
 *    - Located in: response.result.retrieval_assessment.positional_ratings
 *    - Structure: { assessment_name: { rating: [{ position: 0, rating: {...} }] } }
 *
 * @param response - The assessment response from the API
 * @param assessmentName - The name of the assessment (used to flip harmfulness results)
 */
function parseAssessmentResponse(
  response: ChatAssessmentsResponse,
  assessmentName: string,
): {
  assessment_id: string | null;
  result: string | null;
  rationale: string | null;
} {
  // First, try to get response assessment ratings (for non-retrieval assessments)
  const ratings = response.result?.response_assessment?.ratings;

  // Check if this is a retrieval assessment
  const retrievalAssessment = response.result?.retrieval_assessment;
  const positionalRatings = retrievalAssessment?.positional_ratings;

  let firstRating: any = null;

  if (positionalRatings && typeof positionalRatings === 'object') {
    // This is a retrieval assessment - get the assessment by name
    const assessmentData = positionalRatings[assessmentName];
    if (
      assessmentData &&
      assessmentData.rating &&
      Array.isArray(assessmentData.rating) &&
      assessmentData.rating.length > 0
    ) {
      // Get the first position's rating
      firstRating = assessmentData.rating[0].rating;
    }
  } else if (ratings && Object.keys(ratings).length > 0) {
    // This is a response assessment - get the first rating
    firstRating = Object.values(ratings)[0];
  }

  if (!firstRating) {
    return { assessment_id: null, result: null, rationale: null };
  }

  const assessmentId = firstRating.assessment_id || null;
  let value = firstRating.value;
  let rationale = firstRating.rationale;

  if (typeof value === 'object' && value !== null) {
    const structuredValue = value as {
      categorical_value?: string;
      bool_value?: boolean;
      double_value?: number;
      justification?: string;
    };

    // Priority: categorical_value > bool_value > double_value
    value =
      structuredValue.categorical_value ||
      structuredValue.bool_value?.toString() ||
      structuredValue.double_value?.toString() ||
      null;

    rationale = structuredValue.justification || rationale;
  }

  // Flip harmfulness to safety for the Safety judge (yes harmful = no safe, no harmful = yes safe)
  if (assessmentName === 'harmfulness' && typeof value === 'string') {
    const lowerValue = value.toLowerCase();
    if (lowerValue === 'yes') {
      value = 'no';
    } else if (lowerValue === 'no') {
      value = 'yes';
    }
  }

  return {
    assessment_id: assessmentId,
    result: value || null,
    rationale: rationale || null,
  };
}

/**
 * State returned by useEvaluateTraces
 */
export interface EvaluateTracesState {
  data: JudgeEvaluationResult[] | null;
  isLoading: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * React hook for evaluating judges on traces with React Query caching
 *
 * Supports both custom LLM judges (via chat-completions) and built-in judges (via chat-assessments).
 * Results from both endpoints will not be cached since LLM responses are not deterministic.
 */
export function useEvaluateTraces({
  onScorerFinished,
}: {
  /**
   * Callback to be called when the evaluation is finished.
   */
  onScorerFinished?: () => void;
} = {}): [(params: EvaluateTracesParams) => Promise<JudgeEvaluationResult[] | void>, EvaluateTracesState] {
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<JudgeEvaluationResult[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const queryClient = useQueryClient();
  const getTraceIdsForEvaluation = useGetTraceIdsForEvaluation();
  const invocationCounterRef = useRef(0);

  /**
   * Enables asynchronous evaluation. If enabled, the evaluation will be done as a
   * queued job on the server and the results will be polled for asynchronously.
   */
  const usingAsyncMode = isEvaluatingSessionsInScorersEnabled();

  const evaluateTracesSync = useCallback(
    async (params: EvaluateTracesParams): Promise<JudgeEvaluationResult[]> => {
      // Track this invocation to ensure only the latest one calls onScorerFinished
      invocationCounterRef.current += 1;
      const currentInvocationId = invocationCounterRef.current;

      setIsLoading(true);
      setError(null);
      setData(null);

      const { experimentId } = params;

      try {
        // Extract trace IDs from search results
        const traceIds = await getTraceIdsForEvaluation(params);

        // Fetch and evaluate all traces in parallel
        // For traces with multiple retrieval spans, we create multiple results
        const evaluationResultsNested = await Promise.all(
          traceIds.map(async (traceId) => {
            let fullTrace: ModelTrace | null = null;

            try {
              // Fetch trace data with React Query caching
              fullTrace = await queryClient.fetchQuery({
                queryKey: ['GetMlflowTraceV3', traceId],
                queryFn: () => getMlflowTraceV3ForEvaluation(traceId),
                staleTime: Infinity,
                cacheTime: Infinity,
              });

              // Route to appropriate API based on params type
              if (isChatCompletionsParams(params)) {
                // Custom LLM judges path
                const { judgeInstructions } = params;
                const { inputs, outputs, expectations } = extractFromTrace(fullTrace as ModelTrace);

                // Extract template variables from instructions to filter what gets included in user prompt
                const templateVariables = extractTemplateVariables(judgeInstructions);

                if (templateVariables.includes('trace')) {
                  throw new Error('The trace variable is not supported when running the scorer on a sample of traces');
                }

                // Build prompts
                const systemPrompt = buildSystemPrompt(judgeInstructions);
                const userPrompt = buildUserPrompt(inputs, outputs, expectations, templateVariables);

                const response = await callChatCompletions(userPrompt, systemPrompt, experimentId);

                if (response.error_code || response.error_message) {
                  return [
                    {
                      trace: fullTrace,
                      results: [],
                      error: response.error_message || response.error_code || 'Unknown error',
                    },
                  ];
                }

                const { result, rationale } = parseJudgeResponse(response.output);
                return [
                  {
                    trace: fullTrace,
                    results: [
                      {
                        result,
                        rationale,
                        error: null,
                        span_name: '',
                      },
                    ],
                    error: null,
                  },
                ];
              } else {
                // Built-in judges path
                const { requestedAssessments, guidelines } = params;

                // Check if trace is valid
                if (!fullTrace) {
                  return [
                    {
                      trace: null,
                      results: [],
                      error: 'Failed to fetch trace',
                    },
                  ];
                }

                // Extract data from trace
                const chatRequest = extractInputs(fullTrace);
                const chatResponse = extractOutputs(fullTrace);
                const retrievalContexts = extractRetrievalContext(fullTrace);
                const groundTruth = extractExpectations(fullTrace);

                if (!chatRequest) {
                  return [
                    {
                      trace: fullTrace,
                      results: [],
                      error: 'No chat request found in trace',
                    },
                  ];
                }

                // Check if this is a guidelines assessment
                const isGuidelinesAssessment =
                  requestedAssessments.length > 0 && requestedAssessments[0].assessment_name === 'guidelines';

                // Check if this is a retrieval assessment (only these need per-span evaluation)
                const assessmentName = requestedAssessments[0]?.assessment_name || '';
                const isRetrievalAssessment = RETRIEVAL_ASSESSMENTS.includes(assessmentName as any);

                // For retrieval assessments, evaluate each retrieval span separately
                // For non-retrieval assessments, evaluate once without retrieval context
                const retrievalSpans =
                  isRetrievalAssessment && retrievalContexts?.retrieved_documents?.length
                    ? retrievalContexts.retrieved_documents
                    : [null];

                // Evaluate each retrieval span separately (or just once for non-retrieval assessments)
                const spanResults = await Promise.all(
                  retrievalSpans.map(async (retrievalSpan) => {
                    try {
                      const response = await callChatAssessments(
                        chatRequest,
                        chatResponse,
                        retrievalSpan,
                        groundTruth,
                        guidelines || null,
                        requestedAssessments,
                        experimentId,
                        isGuidelinesAssessment,
                      );

                      // Check for errors in the response (both response_assessment and retrieval_assessment)
                      // Check response assessment for errors
                      const ratings = response.result?.response_assessment?.ratings;
                      if (ratings && Object.keys(ratings).length > 0) {
                        const firstRating = Object.values(ratings)[0];
                        if (firstRating?.error) {
                          // Error can be a string or an object with error_code/error_msg
                          let errorMessage: string;
                          if (typeof firstRating.error === 'string') {
                            errorMessage = firstRating.error;
                          } else if (typeof firstRating.error === 'object' && firstRating.error !== null) {
                            const errorObj = firstRating.error as any;
                            errorMessage =
                              errorObj.error_msg || errorObj.error_code || JSON.stringify(firstRating.error);
                          } else {
                            errorMessage = 'Unknown error';
                          }
                          return {
                            trace: fullTrace,
                            result: null,
                            rationale: null,
                            error: errorMessage,
                          };
                        }
                      }

                      // Parse the successful response assessment
                      const { assessment_id, result, rationale } = parseAssessmentResponse(response, assessmentName);
                      return {
                        trace: fullTrace,
                        assessment_id: assessment_id || undefined,
                        result,
                        rationale,
                        error: null,
                        span_name: retrievalSpan?.span_name || '',
                      };
                    } catch (err) {
                      return {
                        trace: fullTrace,
                        result: null,
                        rationale: null,
                        error: err instanceof Error ? err.message : String(err),
                      };
                    }
                  }),
                );

                // Group all span results into a single trace result
                // Check if all results are errors
                const allErrors = spanResults.every((r) => r.error);
                if (allErrors) {
                  // If all spans failed, return the first error
                  return [
                    {
                      trace: fullTrace,
                      results: [],
                      error: spanResults[0].error,
                    },
                  ];
                }

                // Filter out errors and return successful span results
                const successfulResults = spanResults.filter((r) => !r.error);

                // Return a single result with all span results
                return [
                  {
                    trace: fullTrace,
                    results: successfulResults.map((r) => ({
                      result: r.result,
                      rationale: r.rationale,
                      error: r.error,
                      span_name: r.span_name || '',
                    })),
                    error: null,
                  },
                ];
              }
            } catch (err) {
              // Handle both fetch failures and evaluation errors for individual traces
              return [
                {
                  trace: fullTrace,
                  results: [],
                  error: err instanceof Error ? err.message : String(err),
                },
              ];
            }
          }),
        );

        // Flatten results - each trace can produce multiple results (one per retrieval span)
        const evaluationResults: JudgeEvaluationResult[] = evaluationResultsNested.flat();

        setData(evaluationResults);

        // Only call onScorerFinished if this is still the latest invocation
        if (currentInvocationId === invocationCounterRef.current && onScorerFinished) {
          onScorerFinished();
        }

        return evaluationResults;
      } catch (err) {
        const errorObj = err instanceof Error ? err : new Error(String(err));
        setError(errorObj);
        throw errorObj;
      } finally {
        setIsLoading(false);
      }
    },
    [queryClient, getTraceIdsForEvaluation, onScorerFinished],
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setIsLoading(false);
  }, []);

  if (usingAsyncMode) {
    // TODO(next PRs): Return controls and state for async evaluation
  }

  return [
    evaluateTracesSync,
    {
      data,
      isLoading,
      error,
      reset,
    },
  ] as const;
}

/**
 * Parameters for prefetching traces
 */
export interface PrefetchTracesParams {
  traceCount: number;
  locations: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
}

/**
 * React hook for prefetching traces to populate React Query cache
 */
export function usePrefetchTraces({ traceCount, locations }: PrefetchTracesParams): void {
  const queryClient = useQueryClient();
  const isRunningScorersFeatureEnabled = isRunningScorersEnabled();

  useEffect(() => {
    if (!isRunningScorersFeatureEnabled) {
      return;
    }

    const prefetchTraces = async () => {
      try {
        // Fetch the search query to get trace IDs
        const traces = await queryClient.fetchQuery({
          queryKey: [
            SEARCH_MLFLOW_TRACES_QUERY_KEY,
            {
              locations,
              orderBy: ['timestamp DESC'],
              pageSize: traceCount,
            },
          ],
          queryFn: ({ signal }) =>
            searchMlflowTracesQueryFn({
              signal,
              locations,
              pageSize: traceCount,
              limit: traceCount,
              orderBy: ['timestamp DESC'],
            }),
          staleTime: Infinity,
          cacheTime: Infinity,
        });

        // Extract trace IDs
        const traceIds = traces.map((trace) => trace.trace_id).filter((id): id is string => Boolean(id));

        // Prefetch all individual trace get queries (suppress individual errors)
        await Promise.allSettled(
          traceIds.map((traceId) =>
            queryClient.prefetchQuery({
              queryKey: ['GetMlflowTraceV3', traceId],
              queryFn: () => getMlflowTraceV3ForEvaluation(traceId),
              staleTime: Infinity,
              cacheTime: Infinity,
            }),
          ),
        );
      } catch {
        // Silently fail - prefetching is optional and shouldn't cause errors
      }
    };
    prefetchTraces();
  }, [isRunningScorersFeatureEnabled, traceCount, locations, queryClient]);
}
