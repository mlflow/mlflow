import { isNil, isNumber, isPlainObject, orderBy } from 'lodash';

import type { ThemeType } from '@databricks/design-system';
import { CheckCircleIcon, WarningIcon, XCircleIcon } from '@databricks/design-system';
import { defineMessage } from '@databricks/i18n';
import type { MessageDescriptor, IntlShape } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';

import type {
  AssessmentInfo,
  AssessmentValueType,
  RunEvaluationResultAssessment,
  RunEvaluationResultAssessmentDraft,
  RunEvaluationTracesDataEntry,
} from '../types';
import { getEvaluationResultAssessmentBackgroundColor, getEvaluationResultIconColor } from '../utils/Colors';

export const INPUT_REQUEST_KEY = 'request';
const INPUT_MESSAGES_KEY = 'messages';

export enum KnownEvaluationResultAssessmentName {
  OVERALL_ASSESSMENT = 'overall_assessment',
  SAFETY = 'safety',
  GROUNDEDNESS = 'groundedness',
  RETRIEVAL_GROUNDEDNESS = 'retrieval_groundedness', // Updated name for groundedness
  CORRECTNESS = 'correctness',
  RELEVANCE_TO_QUERY = 'relevance_to_query',
  CHUNK_RELEVANCE = 'chunk_relevance',
  RETRIEVAL_RELEVANCE = 'retrieval_relevance', // Updated name for chunk relevance
  CONTEXT_SUFFICIENCY = 'context_sufficiency',
  RETRIEVAL_SUFFICIENCY = 'retrieval_sufficiency', // Updated name for context sufficiency
  GUIDELINE_ADHERENCE = 'guideline_adherence',
  GUIDELINES = 'guidelines', // Updated name for guideline adherence
  GLOBAL_GUIDELINE_ADHERENCE = 'global_guideline_adherence',
}

export const DEFAULT_ASSESSMENTS_SORT_ORDER: string[] = [
  KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
  // Correctness runs with GT, relevancy to query runs without it.
  KnownEvaluationResultAssessmentName.CORRECTNESS,
  KnownEvaluationResultAssessmentName.GLOBAL_GUIDELINE_ADHERENCE,
  KnownEvaluationResultAssessmentName.GUIDELINE_ADHERENCE,
  KnownEvaluationResultAssessmentName.GUIDELINES,
  KnownEvaluationResultAssessmentName.RELEVANCE_TO_QUERY,
  KnownEvaluationResultAssessmentName.CONTEXT_SUFFICIENCY,
  KnownEvaluationResultAssessmentName.RETRIEVAL_SUFFICIENCY,
  KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE,
  KnownEvaluationResultAssessmentName.RETRIEVAL_RELEVANCE,
  KnownEvaluationResultAssessmentName.GROUNDEDNESS,
  KnownEvaluationResultAssessmentName.RETRIEVAL_GROUNDEDNESS, // Updated name for groundedness
  KnownEvaluationResultAssessmentName.SAFETY,
];

export const getJudgeMetricsLink = (asessmentDocLink?: AssessmentLearnMoreLink) => {
  // return OSS docs link
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/';
};

export interface AssessmentLearnMoreLink {
  basePath: string;
  hash?: string;
}

/**
 * These will be converted to the links per-cloud:
 * https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/${hash}`
 * https://docs.databricks.com/en/generative-ai/agent-evaluation/${page}.html#${hash}
 */
export const ASSESSMENTS_DOC_LINKS: Record<string, AssessmentLearnMoreLink> = {
  [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT]: {
    // TODO(nsthorat): Update this link to the overall deep link once it's available.
    basePath: '/generative-ai/agent-evaluation/llm-judge-metrics',
    hash: 'how-quality-is-assessed-by-llm-judges',
  },
  [KnownEvaluationResultAssessmentName.CORRECTNESS]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'correctness',
  },
  [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'groundedness',
  },
  [KnownEvaluationResultAssessmentName.RETRIEVAL_GROUNDEDNESS]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'groundedness',
  },
  [KnownEvaluationResultAssessmentName.RELEVANCE_TO_QUERY]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'answer-relevance',
  },
  [KnownEvaluationResultAssessmentName.SAFETY]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'safety',
  },
  [KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'chunk-relevance-precision',
  },
  [KnownEvaluationResultAssessmentName.RETRIEVAL_RELEVANCE]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'chunk-relevance-precision',
  },
  [KnownEvaluationResultAssessmentName.CONTEXT_SUFFICIENCY]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'context-sufficiency',
  },
  [KnownEvaluationResultAssessmentName.RETRIEVAL_SUFFICIENCY]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'context-sufficiency',
  },
  [KnownEvaluationResultAssessmentName.GUIDELINE_ADHERENCE]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'guideline-adherence',
  },
  [KnownEvaluationResultAssessmentName.GUIDELINES]: {
    basePath: '/generative-ai/agent-evaluation/llm-judge-reference',
    hash: 'guideline-adherence',
  },
};

export enum KnownEvaluationResultAssessmentStringValue {
  YES = 'yes',
  NO = 'no',
  UNKNOWN = 'unknown',
}

export function getAssessmentValueLabel(
  intl: IntlShape,
  theme: ThemeType,
  assessmentInfo: AssessmentInfo,
  value: AssessmentValueType,
): { content: string; icon?: JSX.Element } {
  if (assessmentInfo.dtype === 'pass-fail') {
    if (value === KnownEvaluationResultAssessmentStringValue.YES) {
      return {
        content: intl.formatMessage({
          defaultMessage: 'Pass',
          description: 'Passing assessment label',
        }),
        icon: (
          <span
            css={{
              color: `${getEvaluationResultIconColor(theme, assessmentInfo, {
                stringValue: KnownEvaluationResultAssessmentStringValue.YES,
              })} !important`,
              svg: {
                width: '12px',
                height: '12px',
              },
            }}
          >
            <CheckCircleIcon
              css={{
                backgroundColor: getEvaluationResultAssessmentBackgroundColor(theme, assessmentInfo, {
                  stringValue: KnownEvaluationResultAssessmentStringValue.YES,
                }),
                borderRadius: '50%',
              }}
            />
          </span>
        ),
      };
    } else if (value === KnownEvaluationResultAssessmentStringValue.NO) {
      return {
        content: intl.formatMessage({
          defaultMessage: 'Fail',
          description: 'Failing assessment label',
        }),
        icon: (
          <span
            css={{
              color: `${getEvaluationResultIconColor(theme, assessmentInfo, {
                stringValue: KnownEvaluationResultAssessmentStringValue.NO,
              })} !important`,
              svg: {
                width: '12px',
                height: '12px',
              },
            }}
          >
            <XCircleIcon
              css={{
                backgroundColor: getEvaluationResultAssessmentBackgroundColor(theme, assessmentInfo, {
                  stringValue: KnownEvaluationResultAssessmentStringValue.NO,
                }),
                borderRadius: '50%',
              }}
            />
          </span>
        ),
      };
    } else {
      return {
        content: intl.formatMessage({
          defaultMessage: 'Missing',
          description: 'Missing assessment label',
        }),
        icon: (
          <span
            css={{
              color: `${getEvaluationResultIconColor(theme, assessmentInfo, {
                stringValue: KnownEvaluationResultAssessmentStringValue.UNKNOWN,
              })} !important`,
              svg: {
                width: '12px',
                height: '12px',
              },
            }}
          >
            <WarningIcon
              css={{
                backgroundColor: getEvaluationResultAssessmentBackgroundColor(theme, assessmentInfo, {
                  stringValue: KnownEvaluationResultAssessmentStringValue.UNKNOWN,
                }),
                borderRadius: '50%',
              }}
            />
          </span>
        ),
      };
    }
  } else if (assessmentInfo.dtype === 'boolean') {
    if (value === true) {
      return {
        content: intl.formatMessage({
          defaultMessage: 'True',
          description: 'True assessment label',
        }),
      };
    } else if (value === false) {
      return {
        content: intl.formatMessage({
          defaultMessage: 'False',
          description: 'False assessment label',
        }),
      };
    } else {
      return {
        content: intl.formatMessage({
          defaultMessage: 'null',
          description: 'null assessment label',
        }),
      };
    }
  }
  return {
    content: `${value}`,
  };
}

export enum KnownEvaluationResultAssessmentMetadataFields {
  IS_OVERALL_ASSESSMENT = 'is_overall_assessment',
  CHUNK_INDEX = 'chunk_index',
  IS_COPIED_FROM_AI = 'is_copied_from_ai',
}

export const KnownEvaluationResultAssessmentOutputLabel: Record<string, MessageDescriptor> = {
  response: defineMessage({
    defaultMessage: 'Model output',
    description: 'Evaluation review > Response section > model output > title',
  }),
};

export const EXPECTED_FACTS_FIELD_NAME = 'expected_facts';

export const KnownEvaluationResultAssessmentTargetLabel: Record<string, MessageDescriptor> = {
  expected_response: defineMessage({
    defaultMessage: 'Expected output',
    description: 'Evaluation review > Response section > expected output > title',
  }),
  [EXPECTED_FACTS_FIELD_NAME]: defineMessage({
    defaultMessage: 'Expected facts',
    description: 'Evaluation review > Response section > expected facts > title',
  }),
};

export const KnownEvaluationResultAssessmentValueLabel: Record<string, MessageDescriptor> = {
  [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT]: defineMessage({
    defaultMessage: 'Overall',
    description: 'Evaluation results > known type of evaluation result assessment > overall assessment.',
  }),
  [KnownEvaluationResultAssessmentName.CORRECTNESS]: defineMessage({
    defaultMessage: 'Correctness',
    description:
      'Evaluation results > known type of evaluation result assessment > correctness assessment. Used to indicate if the result is correct in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Correctness: mostly yes, have gaps"',
  }),
  [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: defineMessage({
    defaultMessage: 'Groundedness',
    description:
      'Evaluation results > known type of evaluation result assessment > groundedness assessment. Used to indicate if the result is grounded in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Groundedness: moderately grounded"',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_GROUNDEDNESS]: defineMessage({
    defaultMessage: 'Retrieval groundedness',
    description:
      'Evaluation results > known type of evaluation result assessment > retrieval groundedness assessment. Used to indicate if the result is grounded in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Retrieval groundedness: moderately grounded"',
  }),
  [KnownEvaluationResultAssessmentName.CONTEXT_SUFFICIENCY]: defineMessage({
    defaultMessage: 'Context sufficiency',
    description:
      'Evaluation results > known type of evaluation result assessment > context sufficiency assessment. Used to indicate if the retrieved context is sufficient to generate the expected response. Label displayed if user provided custom value, e.g. "Context Sufficiency: mostly sufficient"',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_SUFFICIENCY]: defineMessage({
    defaultMessage: 'Retrieval sufficiency',
    description:
      'Evaluation results > known type of evaluation result assessment > retrieval sufficiency assessment. Used to indicate if the retrieved context is sufficient to generate the expected response. Label displayed if user provided custom value, e.g. "Retrieval sufficiency: mostly sufficient"',
  }),
  [KnownEvaluationResultAssessmentName.RELEVANCE_TO_QUERY]: defineMessage({
    defaultMessage: 'Relevance',
    description:
      'Evaluation results > known type of evaluation result assessment > relevance assessment. Used to indicate if the result is relevant to query in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Relevance: moderate"',
  }),
  [KnownEvaluationResultAssessmentName.SAFETY]: defineMessage({
    defaultMessage: 'Safety',
    description:
      'Evaluation results > known type of evaluation result assessment > safety assessment. Used to indicate if the result is safe in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Safety: moderate"',
  }),
  [KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE]: defineMessage({
    defaultMessage: 'Chunk relevance',
    description:
      'Evaluation results > known type of evaluation result assessment > chunk relevance assessment. Used to indicate if the result is relevant to source data chunk in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Chunk relevance: moderate"',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_RELEVANCE]: defineMessage({
    defaultMessage: 'Retrieval relevance',
    description:
      'Evaluation results > known type of evaluation result assessment > retrieval relevance assessment. Used to indicate if the result is relevant to source data chunk in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Retrieval relevance: moderate"',
  }),
  [KnownEvaluationResultAssessmentName.GUIDELINE_ADHERENCE]: defineMessage({
    defaultMessage: 'Guideline adherence',
    description:
      'Evaluation results > known type of evaluation result assessment > guideline adherence assessment. Used to indicate if the result adheres to the guidelines in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Guideline adherence: moderate"',
  }),
  [KnownEvaluationResultAssessmentName.GUIDELINES]: defineMessage({
    defaultMessage: 'Guidelines',
    description:
      'Evaluation results > known type of evaluation result assessment > guidelines assessment. Used to indicate if the result adheres to the guidelines in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Guidelines: moderate"',
  }),
  [KnownEvaluationResultAssessmentName.GLOBAL_GUIDELINE_ADHERENCE]: defineMessage({
    defaultMessage: 'Global guideline adherence',
    description:
      'Evaluation results > known type of evaluation result assessment > global guideline adherence assessment. Used to indicate if the result adheres to the global guidelines in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Global guideline adherence: moderate"',
  }),
};

export const KnownEvaluationResultAssessmentValueMissingTooltip: Record<string, MessageDescriptor> = {
  [KnownEvaluationResultAssessmentName.CORRECTNESS]: defineMessage({
    defaultMessage:
      'Correctness assessment is missing. This is likely because you have not provided an expected response (ground truth) or grading notes.',
    description:
      'Evaluation results > known type of evaluation result assessment > correctness assessment. Used to indicate if the result is correct in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Correctness: mostly yes, have gaps"',
  }),
  [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: defineMessage({
    defaultMessage:
      'Groundedness assessment is missing. This is likely because your agent is not returning retrieved_context.',
    description:
      'Evaluation results > known type of evaluation result assessment > groundedness assessment. Used to indicate if the result is grounded in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Groundedness: moderately grounded"',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_GROUNDEDNESS]: defineMessage({
    defaultMessage:
      'Retrieval Groundedness assessment is missing. This is likely because your agent is not returning retrieved_context.',
    description:
      'Evaluation results > known type of evaluation result assessment > retrieval groundedness assessment. Used to indicate if the result is grounded in context of LLMs evaluation. Label displayed if user provided custom value, e.g. "Retrieval Groundedness: moderately grounded"',
  }),
  [KnownEvaluationResultAssessmentName.CONTEXT_SUFFICIENCY]: defineMessage({
    defaultMessage:
      'Context sufficiency assessment is missing. This is likely because your agent is not returning retrieved_context.',
    description:
      'Evaluation results > known type of evaluation result assessment > context sufficiency assessment. Used to indicate if the retrieved context is sufficient to generate the expected response. Label displayed if user provided custom value, e.g. "Context Sufficiency: mostly sufficient"',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_SUFFICIENCY]: defineMessage({
    defaultMessage:
      'Retrieval sufficiency assessment is missing. This is likely because your agent is not returning retrieved_context.',
    description:
      'Evaluation results > known type of evaluation result assessment > retrieval sufficiency assessment. Used to indicate if the retrieved context is sufficient to generate the expected response. Label displayed if user provided custom value, e.g. "Retrieval sufficiency: mostly sufficient"',
  }),
};

export const KnownEvaluationResultAssessmentValueDescription: Record<string, MessageDescriptor> = {
  [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT]: defineMessage({
    defaultMessage: 'The overall assessment passes when all of the judges pass.',
    description:
      'Evaluation results > known type of evaluation result assessment > overall assessment > description of overall assessment',
  }),
  [KnownEvaluationResultAssessmentName.CORRECTNESS]: defineMessage({
    defaultMessage:
      "The correctness LLM judge gives a binary evaluation and written rationale on whether the agent's generated response is factually accurate and semantically similar to the provided ground-truth response or grading notes.",
    description:
      'Evaluation results > known type of evaluation result assessment > description of correctness assessment.',
  }),
  [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: defineMessage({
    defaultMessage:
      "The groundedness LLM judge returns a binary evaluation and written rationale on whether the generated response is factually consistent with the agent's retrieved context.",
    description: 'Evaluation results > known type of evaluation result assessment > description of groundedness judge.',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_GROUNDEDNESS]: defineMessage({
    defaultMessage:
      "The retrieval groundedness LLM judge returns a binary evaluation and written rationale on whether the generated response is factually consistent with the agent's retrieved context.",
    description:
      'Evaluation results > known type of evaluation result assessment > description of retrieval groundedness judge.',
  }),
  [KnownEvaluationResultAssessmentName.RELEVANCE_TO_QUERY]: defineMessage({
    defaultMessage:
      'The relevance_to_query LLM judge determines whether the response is relevant to the input request.',
    description: 'Evaluation results > known type of evaluation result assessment > description of relevance judge.',
  }),
  [KnownEvaluationResultAssessmentName.SAFETY]: defineMessage({
    defaultMessage:
      'The safety LLM judge returns a binary rating and a written rationale on whether the generated response has harmful or toxic content.',
    description: 'Evaluation results > known type of evaluation result assessment > description of safety judge.',
  }),
  [KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE]: defineMessage({
    defaultMessage:
      'The chunk-relevance-precision LLM judge determines whether the chunks returned by the retriever are relevant to the input request. Precision is calculated as the number of relevant chunks returned divided by the total number of chunks returned. For example, if the retriever returns four chunks, and the LLM judge determines that three of the four returned documents are relevant to the request, then llm_judged/chunk_relevance/precision is 0.75.',
    description: 'Evaluation results > known type of evaluation result assessment > chunk relevance judge.',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_RELEVANCE]: defineMessage({
    defaultMessage:
      'The retrieval-relevance LLM judge determines whether the chunks returned by the retriever are relevant to the input request. Precision is calculated as the number of relevant chunks returned divided by the total number of chunks returned. For example, if the retriever returns four chunks, and the LLM judge determines that three of the four returned documents are relevant to the request, then llm_judged/chunk_relevance/precision is 0.75.',
    description: 'Evaluation results > known type of evaluation result assessment > retrieval relevance judge.',
  }),
  [KnownEvaluationResultAssessmentName.CONTEXT_SUFFICIENCY]: defineMessage({
    defaultMessage:
      'The context sufficiency LLM judge determines whether the retrieved context is sufficient to generate the expected response.',
    description: 'Evaluation results > known type of evaluation result assessment > context sufficiency judge.',
  }),
  [KnownEvaluationResultAssessmentName.RETRIEVAL_SUFFICIENCY]: defineMessage({
    defaultMessage:
      'The retrieval sufficiency LLM judge determines whether the retrieved context is sufficient to generate the expected response.',
    description: 'Evaluation results > known type of evaluation result assessment > retrieval sufficiency judge.',
  }),
  [KnownEvaluationResultAssessmentName.GUIDELINE_ADHERENCE]: defineMessage({
    defaultMessage:
      'The guideline adherence LLM judge determines whether the response adheres to the guidelines provided.',
    description: 'Evaluation results > known type of evaluation result assessment > guideline adherence judge.',
  }),
  [KnownEvaluationResultAssessmentName.GUIDELINES]: defineMessage({
    defaultMessage:
      'The guidelines LLM judge determines whether the response adheres to the guidelines provided. All responses must adhere to guidelines.',
    description: 'Evaluation results > known type of evaluation result assessment > guidelines judge.',
  }),
  [KnownEvaluationResultAssessmentName.GLOBAL_GUIDELINE_ADHERENCE]: defineMessage({
    defaultMessage:
      'The global guideline adherence LLM judge determines whether the response adheres to the global guidelines provided. All responses must adhere to global guidelines.',
    description: 'Evaluation results > known type of evaluation result assessment > global guideline adherence judge.',
  }),
};

export const KnownEvaluationResultAssessmentValueMapping: Record<string, Record<string, MessageDescriptor>> = {
  [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Pass',
      description:
        'Evaluation results > overall assessment > pass value label. Displayed if evaluation result is overall considered as approved by LLM judge or human.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Fail',
      description:
        'Evaluation results > overall assessment > fail value label. Displayed if evaluation result is overall considered as disapproved by LLM judge or human.',
    }),
  },
  [KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Relevant',
      description:
        'Evaluation results > chunk relevancy assessment > positive value label. Displayed if evaluation result is considered as relevant to the query.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Irrelevant',
      description:
        'Evaluation results > chunk relevancy assessment > negative value label. Displayed if evaluation result is considered as irrelevant to the query.',
    }),
  },
  [KnownEvaluationResultAssessmentName.RETRIEVAL_RELEVANCE]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Relevant',
      description:
        'Evaluation results > retrieval relevancy assessment > positive value label. Displayed if evaluation result is considered as relevant to the query.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Irrelevant',
      description:
        'Evaluation results > retrieval relevancy assessment > negative value label. Displayed if evaluation result is considered as irrelevant to the query.',
    }),
  },
  [KnownEvaluationResultAssessmentName.CONTEXT_SUFFICIENCY]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Context sufficient',
      description:
        'Evaluation results > context sufficiency assessment > positive value label. Displayed if retrieved context is sufficient to generate the expected response.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Context insufficient',
      description:
        'Evaluation results > context sufficiency assessment > negative value label. Displayed if retrieved context is insufficient to generate the expected response.',
    }),
  },
  [KnownEvaluationResultAssessmentName.RETRIEVAL_SUFFICIENCY]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Sufficient',
      description:
        'Evaluation results > retrieval sufficiency assessment > positive value label. Displayed if retrieved context is sufficient to generate the expected response.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Insufficient',
      description:
        'Evaluation results > retrieval sufficiency assessment > negative value label. Displayed if retrieved context is insufficient to generate the expected response.',
    }),
  },
  [KnownEvaluationResultAssessmentName.RELEVANCE_TO_QUERY]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Relevant',
      description:
        'Evaluation results > relevancy assessment > positive value label. Displayed if evaluation result is considered as irrelevant to the query.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Irrelevant',
      description:
        'Evaluation results > relevancy assessment > negative value label. Displayed if evaluation result is considered as irrelevant to the query.',
    }),
  },
  [KnownEvaluationResultAssessmentName.GROUNDEDNESS]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Grounded',
      description:
        'Evaluation results > grounded assessment > positive value label. Displayed if evaluation result is considered as grounded.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Not grounded',
      description:
        'Evaluation results > grounded assessment > negative value label. Displayed if evaluation result is considered as not grounded.',
    }),
  },
  [KnownEvaluationResultAssessmentName.RETRIEVAL_GROUNDEDNESS]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Grounded',
      description:
        'Evaluation results > retrieval grounded assessment > positive value label. Displayed if evaluation result is considered as grounded.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Not grounded',
      description:
        'Evaluation results > retrieval grounded assessment > negative value label. Displayed if evaluation result is considered as not grounded.',
    }),
  },
  [KnownEvaluationResultAssessmentName.CORRECTNESS]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Correct',
      description:
        'Evaluation results > correctness assessment > positive value label. Displayed if evaluation result is considered as correct.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Incorrect',
      description:
        'Evaluation results > correctness assessment > negative value label. Displayed if evaluation result is considered as incorrect.',
    }),
  },
  [KnownEvaluationResultAssessmentName.SAFETY]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Safe',
      description:
        'Evaluation results > safety assessment > positive value label. Displayed if evaluation result is considered as safe.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Unsafe',
      description:
        'Evaluation results > safety assessment > negative value label. Displayed if evaluation result is considered as unsafe.',
    }),
  },
  [KnownEvaluationResultAssessmentName.GUIDELINE_ADHERENCE]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Adheres to guidelines',
      description:
        'Evaluation results > guideline adherence assessment > positive value label. Displayed if evaluation result adheres to the guidelines.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Violates guidelines',
      description:
        'Evaluation results > guideline adherence assessment > negative value label. Displayed if evaluation result does not adhere to the guidelines.',
    }),
  },
  [KnownEvaluationResultAssessmentName.GUIDELINES]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Adheres to guidelines',
      description:
        'Evaluation results > guidelines assessment > positive value label. Displayed if evaluation result adheres to the guidelines.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Violates guidelines',
      description:
        'Evaluation results > guidelines assessment > negative value label. Displayed if evaluation result does not adhere to the guidelines.',
    }),
  },
  [KnownEvaluationResultAssessmentName.GLOBAL_GUIDELINE_ADHERENCE]: {
    [KnownEvaluationResultAssessmentStringValue.YES]: defineMessage({
      defaultMessage: 'Adheres to global guidelines',
      description:
        'Evaluation results > global guideline adherence assessment > positive value label. Displayed if evaluation result adheres to the global guidelines.',
    }),
    [KnownEvaluationResultAssessmentStringValue.NO]: defineMessage({
      defaultMessage: 'Violates global guidelines',
      description:
        'Evaluation results > global guideline adherence assessment > negative value label. Displayed if evaluation result does not adhere to the global guidelines.',
    }),
  },
};

const isAssessmentAiGenerated = (assessment: RunEvaluationResultAssessment) => {
  return assessment?.source?.sourceType === 'AI_JUDGE';
};

/**
 * Returns the title for the given evaluation result.
 * If the evaluation has an input request, it will be used as the title. Otherwise, the evaluation ID will be used.
 * Stringifies the value if it's an object or an array.
 */
export const getEvaluationResultTitle = (evaluation: RunEvaluationTracesDataEntry): string => {
  // Use the request as the title if defined.
  let title = getEvaluationResultInputTitle(evaluation, INPUT_REQUEST_KEY);

  if (isNil(title)) {
    title = getEvaluationResultInputTitle(evaluation, INPUT_MESSAGES_KEY);
  }

  // If the title is still undefined, JSON-serialize the inputs.
  if (isNil(title) && !isNil(evaluation.inputs) && Object.keys(evaluation.inputs).length > 0) {
    title = stringifyValue(evaluation.inputs);
  }

  if (isNil(title) || title === '') {
    title = evaluation.evaluationId;
  }

  return title;
};

/**
 * Returns the title for the given evaluation result and input key.
 * This is different than getEvaluationResultTitle in that it computes a title per input key. getEvaluationResultTitle returns a title for the
 * whole row (used in the header of an evaluation modal).
 */
export const getEvaluationResultInputTitle = (
  evaluation: RunEvaluationTracesDataEntry,
  inputKey: string,
): string | undefined => {
  if (!isNil(evaluation.inputsTitle)) {
    return typeof evaluation.inputsTitle === 'string' ? evaluation.inputsTitle : JSON.stringify(evaluation.inputsTitle);
  }
  // Use the request as the title if defined.
  let title: string | undefined = undefined;
  // Use the last message content as the title if defined.
  const input = evaluation.inputs[inputKey];
  if (
    isPlainObject(input) &&
    !isNil(input[INPUT_MESSAGES_KEY]) &&
    Array.isArray(input[INPUT_MESSAGES_KEY]) &&
    !isNil(input[INPUT_MESSAGES_KEY][0]?.content)
  ) {
    title = input[INPUT_MESSAGES_KEY][input[INPUT_MESSAGES_KEY].length - 1]?.content;
  } else if (!isNil(input) && Array.isArray(input) && !isNil(input[0]?.content)) {
    // Try to parse OpenAI messages.
    title = input[input.length - 1]?.content;
  } else {
    title = input ? stringifyValue(input) : undefined;
  }

  return title;
};

export const isEvaluationResultOverallAssessment = (assessmentEntry: RunEvaluationResultAssessment) =>
  assessmentEntry.metadata?.[KnownEvaluationResultAssessmentMetadataFields.IS_OVERALL_ASSESSMENT] === true;

export const isEvaluationResultPerRetrievalChunkAssessment = (assessmentEntry: RunEvaluationResultAssessment) =>
  isNumber(assessmentEntry.metadata?.[KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX]);

export const getEvaluationResultAssessmentValue = (
  assessment: RunEvaluationResultAssessment,
): AssessmentValueType | undefined => {
  const value = assessment.stringValue ?? assessment.numericValue ?? assessment.booleanValue;
  if (isNil(value)) {
    return undefined;
  }
  return value;
};

/**
 * Add alpha channel to the given hex color.
 */
export function withAlpha(hexColor: string, opacity: number): string {
  let color = hexColor;
  const startsWithHash = color.startsWith('#');
  if (startsWithHash) {
    color = hexColor.slice(1);
  }
  const alpha = Math.round(Math.min(Math.max(opacity, 0), 1) * 255);
  const hexAlpha = alpha.toString(16).toUpperCase();
  return `${startsWithHash ? '#' : ''}${color}${hexAlpha}`;
}

/**
 * A list of known response assessment names, used to populate suggestions
 */
export const KnownEvaluationResponseAssessmentNames = [
  KnownEvaluationResultAssessmentName.GUIDELINE_ADHERENCE,
  KnownEvaluationResultAssessmentName.GUIDELINES,
  KnownEvaluationResultAssessmentName.GLOBAL_GUIDELINE_ADHERENCE,
  KnownEvaluationResultAssessmentName.RELEVANCE_TO_QUERY,
  KnownEvaluationResultAssessmentName.CONTEXT_SUFFICIENCY,
  KnownEvaluationResultAssessmentName.RETRIEVAL_SUFFICIENCY,
  KnownEvaluationResultAssessmentName.CORRECTNESS,
  KnownEvaluationResultAssessmentName.GROUNDEDNESS,
  KnownEvaluationResultAssessmentName.RETRIEVAL_GROUNDEDNESS,
  KnownEvaluationResultAssessmentName.SAFETY,
];

/**
 * A list of known retrieval assessment names, used to populate suggestions
 */
export const KnownEvaluationRetrievalAssessmentNames = [KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE];

/**
 * Creates a draft assessment object with the given values.
 */
export const createDraftEvaluationResultAssessmentObject = ({
  name,
  isOverallAssessment,
  value,
  rationale,
  metadata = {},
}: {
  isOverallAssessment: boolean;
  name: string;
  value: string | boolean;
  rationale?: string;
  metadata?: RunEvaluationResultAssessment['metadata'];
}): RunEvaluationResultAssessmentDraft => {
  // Use current user's email to set as source ID
  const sourceId = getUser() ?? '';

  const resultMetadata = isOverallAssessment
    ? { ...metadata, [KnownEvaluationResultAssessmentMetadataFields.IS_OVERALL_ASSESSMENT]: true }
    : metadata;

  const booleanValue = typeof value === 'boolean' ? value : null;
  const stringValue = typeof value !== 'boolean' ? value : null;

  return {
    booleanValue: booleanValue,
    numericValue: null,
    stringValue: stringValue,
    name,
    metadata: resultMetadata,
    rationale: rationale ?? null,
    source: {
      sourceId,
      sourceType: 'HUMAN',
      metadata: {},
    },
    timestamp: Date.now(),
    isDraft: true,
  };
};

export const shouldRepeatExistingOriginalOverallAiAssessment = (
  sourceEvaluationResult: RunEvaluationTracesDataEntry,
  pendingAssessmentEntries: RunEvaluationResultAssessmentDraft[],
) =>
  sourceEvaluationResult.overallAssessments.length > 0 &&
  sourceEvaluationResult.overallAssessments.every(isAssessmentAiGenerated) &&
  !pendingAssessmentEntries.some(isEvaluationResultOverallAssessment);

export const copyAiOverallAssessmentAsHumanAssessment = (
  sourceEvaluationResult: RunEvaluationTracesDataEntry,
): RunEvaluationResultAssessmentDraft | null => {
  const firstAiOverallAssessment = sourceEvaluationResult.overallAssessments.find(isAssessmentAiGenerated);

  if (!firstAiOverallAssessment) {
    return null;
  }

  const sourceId = getUser() ?? '';

  return {
    ...firstAiOverallAssessment,
    timestamp: Date.now(),
    isDraft: true,
    source: {
      sourceType: 'HUMAN',
      sourceId,
      metadata: {},
    },
    metadata: {
      ...firstAiOverallAssessment.metadata,
      // Explicitly marking it as reviewed to indicate this assessment is copied from AI
      [KnownEvaluationResultAssessmentMetadataFields.IS_COPIED_FROM_AI]: true,
    },
  };
};

export const isDraftAssessment = (
  assessment: RunEvaluationResultAssessment | RunEvaluationResultAssessmentDraft,
): assessment is RunEvaluationResultAssessmentDraft => 'isDraft' in assessment && assessment.isDraft;

/**
 * Returns a list of detailed assessments.
 *
 * For well-known assessments, the list is sorted based on the known stable order;
 * for other assessments, the list is sorted based on the timestamp of the first appearance
 * of the assessment (last item in the group).
 */
export const getOrderedAssessments = (assessmentsByName: Record<string, RunEvaluationResultAssessment[]>) =>
  orderBy(Object.entries(assessmentsByName), ([key, assessments], index) => {
    // If we're dealing with a known detailed assessment, we want to sort it based on its index in the known names list
    // so its position is stable
    const indexInKnownNames = DEFAULT_ASSESSMENTS_SORT_ORDER.indexOf(key as KnownEvaluationResultAssessmentName);

    if (indexInKnownNames !== -1) {
      // If it's a known detailed assessment, sort by its index in the known names list
      return indexInKnownNames;
    } else {
      // Otherwise, sort by the timestamp of the last item in the group
      return assessments[assessments.length - 1]?.timestamp ?? index;
    }
  });

export const isEvaluationResultReviewedAlready = (evaluationResult: RunEvaluationTracesDataEntry) =>
  evaluationResult.overallAssessments
    ?.filter((assessment) => !isDraftAssessment(assessment))
    .some((assessment) => !isAssessmentAiGenerated(assessment)) ?? false;

export const hasBeenEditedByHuman = (assessment: RunEvaluationResultAssessment) =>
  // It is not AI generated, and it doesn't have the `IS_FROM_AI` metadata field set to true
  assessment.source?.sourceType === 'HUMAN' &&
  !assessment.metadata?.[KnownEvaluationResultAssessmentMetadataFields.IS_COPIED_FROM_AI];

export const getEvaluationResultAssessmentChunkIndex = (assessment: RunEvaluationResultAssessment) =>
  assessment.metadata?.[KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX];

// Auto select the first non-empty evaluation ID if no evaluation ID is selected
export const autoSelectFirstNonEmptyEvaluationId = (
  evaluationResults: RunEvaluationTracesDataEntry[] | null,
  selectedEvaluationId: string | undefined,
  setSelectedEvaluationId: (evaluationId: string | undefined) => void,
) => {
  if (!selectedEvaluationId && evaluationResults) {
    // Find first non-empty evaluationId in data
    const firstNonEmpty = evaluationResults.find((evaluation) => evaluation.evaluationId);
    if (firstNonEmpty) {
      setSelectedEvaluationId(firstNonEmpty.evaluationId);
    }
  }
};

/**
 * Converts the given value to a string if it's an object or an array.
 */
export const stringifyValue = (value: any) => {
  return isPlainObject(value) || Array.isArray(value) ? JSON.stringify(value, undefined, 2) : value;
};

/**
 * Utility function: generates suggestions for the assessment values based on original assessment and options.
 */
export const getAssessmentValueSuggestions = (
  intl: IntlShape,
  originalAssessment?: RunEvaluationResultAssessment,
  assessmentHistory?: RunEvaluationResultAssessment[],
  assessmentInfos?: AssessmentInfo[],
) => {
  // If we're starting with an existing assessment, we should suggest the values that are relevant to it.
  if (originalAssessment) {
    const mapping = KnownEvaluationResultAssessmentValueMapping[originalAssessment.name];
    if (!mapping) {
      return [];
    }

    return Object.entries(mapping).map(([key, value]) => {
      return { key, label: intl.formatMessage(value), rootAssessmentName: originalAssessment.name };
    });
  }

  // If we're starting with a new assessment, we only suggest 'boolean' values and a new value.
  return (assessmentInfos || [])
    .filter((assessmentInfo) => assessmentInfo.dtype === 'boolean')
    .map((assessmentInfo) => ({
      key: assessmentInfo.name,
      label: assessmentInfo.name,
      rootAssessmentName: assessmentInfo.name,
      // Disabled when the assessment already exists.
      disabled: assessmentHistory?.some((assessment) => assessment.name === assessmentInfo.name),
    }));
};

/**
 * Returns true if the assessment is missing.
 * An assessment is considered missing if it doesn't have a value, rationale or error message.
 */
export const isAssessmentMissing = (assessment?: RunEvaluationResultAssessment) => {
  if (!assessment) {
    return false;
  } else {
    const hasRationale = Boolean(assessment.rationale);
    const hasValue = !isNil(getEvaluationResultAssessmentValue(assessment));
    const hasErrorMessage = Boolean(assessment.errorMessage);
    return !(hasRationale || hasValue || hasErrorMessage);
  }
};

/**
 * Checks if the given value is a retrieved context.
 * A retrieved context is a list of objects with a `doc_uri` and `content` field.
 */
export const isRetrievedContext = (value: any): boolean => {
  return Array.isArray(value) && value.every((v) => isPlainObject(v) && 'doc_uri' in v && 'content' in v);
};
