import { parseOpenAIAnswer, resolveOpenAIDecision } from './openai';
import { parseTypeSafeAnswer, resolveTypeSafeDecision } from './typesafe';
import type { Answer, Decision, DecisionSpan, Entry } from './shared';

export const resolveDecision = (span?: DecisionSpan | null): Decision | null => {
  switch (span?.chatMessageFormat) {
    case 'typesafe':
      return resolveTypeSafeDecision(span);
    case 'openai_decisions':
      return resolveOpenAIDecision(span);
    default:
      return null;
  }
};

export const parseAnswer = (entry: Entry, source: Decision['source']): Answer =>
  source === 'openai_decisions' ? parseOpenAIAnswer(entry) : parseTypeSafeAnswer(entry);
