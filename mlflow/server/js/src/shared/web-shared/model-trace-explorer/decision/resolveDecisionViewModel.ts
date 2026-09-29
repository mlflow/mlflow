import type { DecisionSpanLike, DecisionViewModel } from './Decision.types';
import { decisionTranslators } from './translators';

export const resolveDecisionViewModel = (span?: DecisionSpanLike | null): DecisionViewModel | null => {
  if (!span) {
    return null;
  }

  for (const translator of decisionTranslators) {
    if (translator.matches(span)) {
      return translator.translate(span);
    }
  }

  return null;
};
