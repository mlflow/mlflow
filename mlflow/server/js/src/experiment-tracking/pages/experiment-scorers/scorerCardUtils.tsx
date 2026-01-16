import React from 'react';
import { SparkleDoubleIcon, CodeIcon, CheckCircleIcon, XCircleIcon } from '@databricks/design-system';
import type { TagColors } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';
import type { UseFormReset } from 'react-hook-form';
import { isNil } from 'lodash';
import type { ScheduledScorer, LLMScorer, CustomCodeScorer } from './types';
import type { LLMScorerFormData } from './LLMScorerFormRenderer';
import type { CustomCodeScorerFormData } from './CustomCodeScorerFormRenderer';
import { TEMPLATE_INSTRUCTIONS_MAP } from './prompts';
import { ScorerFormData, outputTypeSpecToFormData } from './utils/scorerTransformUtils';
import { ScorerEvaluationScope } from './constants';

export const getTypeDisplayName = (scorer: ScheduledScorer, intl: IntlShape): string => {
  if (scorer.type === 'custom-code') {
    return intl.formatMessage({
      defaultMessage: 'Custom code',
      description: 'Label for custom code scorer type',
    });
  }
  if (scorer.type === 'llm') {
    return intl.formatMessage({
      defaultMessage: 'LLM-as-a-judge',
      description: 'Label for LLM scorer type',
    });
  }
  throw new Error(`Unknown scorer type ${(scorer as any).type}`);
};

export const getTypeIcon = (scorer: ScheduledScorer): React.ReactNode => {
  if (scorer.type === 'custom-code') {
    return <CodeIcon />;
  }
  if (scorer.type === 'llm') {
    return <SparkleDoubleIcon />;
  }
  throw new Error(`Unknown scorer type ${(scorer as any).type}`);
};

export const getTypeColor = (scorer: ScheduledScorer): 'pink' | 'purple' => {
  if (scorer.type === 'custom-code') {
    return 'pink';
  }
  return 'purple';
};

const isActive = (scorer: ScheduledScorer): boolean => {
  if (isNil(scorer.sampleRate)) return false;
  return scorer.sampleRate > 0;
};

export const getStatusTag = (
  scorer: ScheduledScorer,
  intl: IntlShape,
): { text: string; color: TagColors; icon: React.ReactNode } => {
  const active = isActive(scorer);
  return {
    text: active
      ? intl.formatMessage({
          defaultMessage: 'Evaluating traces: ON',
          description: 'Status label for active scorer',
        })
      : intl.formatMessage({
          defaultMessage: 'Evaluating traces: OFF',
          description: 'Status label for stopped scorer',
        }),
    color: active ? 'lime' : 'pink',
    icon: active ? <CheckCircleIcon /> : <XCircleIcon />,
  };
};

/**
 * Helper function to derive form values from a ScheduledScorer
 * @param scorer The ScheduledScorer to derive form values from
 * @returns Form values object suitable for react-hook-form
 */
export const getFormValuesFromScorer = (scorer: ScheduledScorer): LLMScorerFormData | ScorerFormData => {
  // For LLM scorers, get instructions from the scorer or fall back to template defaults
  let instructions = '';
  if (scorer.type === 'llm') {
    const llmScorer = scorer as LLMScorer;
    // Use stored instructions if available, otherwise look up from template map
    const templateInstructions = llmScorer.llmTemplate ? TEMPLATE_INSTRUCTIONS_MAP[llmScorer.llmTemplate] : '';
    instructions = llmScorer.instructions || templateInstructions || '';
  }

  const outputTypeFormFields = scorer.type === 'llm' ? outputTypeSpecToFormData((scorer as LLMScorer).outputType) : {};

  return {
    llmTemplate: scorer.type === 'llm' ? (scorer as LLMScorer).llmTemplate || '' : '',
    name: scorer.name || '',
    sampleRate: scorer.sampleRate || 0,
    code: scorer.type === 'custom-code' ? (scorer as CustomCodeScorer).code || '' : '',
    scorerType: scorer.type,
    guidelines: scorer.type === 'llm' ? (scorer as LLMScorer).guidelines?.join('\n') || '' : '',
    instructions,
    filterString: scorer.filterString || '',
    model: scorer.type === 'llm' ? (scorer as LLMScorer).model || '' : '',
    disableMonitoring: scorer.disableMonitoring,
    isInstructionsJudge: scorer.type === 'llm' ? (scorer as LLMScorer).is_instructions_judge : undefined,
    evaluationScope: scorer.isSessionLevelScorer ? ScorerEvaluationScope.SESSIONS : ScorerEvaluationScope.TRACES,
    ...outputTypeFormFields,
  };
};

/**
 * Helper function to sync form state with scorer prop values
 * @param scorer The ScheduledScorer to derive form values from
 * @param reset The react-hook-form reset function
 */
export const syncFormWithScorer = (
  scorer: ScheduledScorer,
  reset: UseFormReset<LLMScorerFormData | CustomCodeScorerFormData>,
): void => {
  const formValues = getFormValuesFromScorer(scorer);
  reset(formValues);
};
