import type { GuardrailStage } from '../../types';

export const STAGE_HINTS: Record<GuardrailStage, string> = {
  BEFORE: 'Receives {{ inputs }} (the incoming request). Example: "Is {{ inputs }} free of profanity?"',
  AFTER:
    'Receives {{ inputs }} (the request) and {{ outputs }} (the response). Example: "Does {{ outputs }} correctly answer {{ inputs }}?"',
};

/** Returns an error message if instructions are incompatible with the stage, or null if valid. */
export const validateStageInstructions = (instructions: string, stage: GuardrailStage): string | null => {
  if (!instructions.trim()) return null;
  if (stage === 'BEFORE') {
    if (instructions.includes('{{ outputs }}')) {
      return '{{ outputs }} is not available in BEFORE stage — the LLM has not run yet';
    }
    return instructions.includes('{{ inputs }}') ? null : 'BEFORE-stage instructions must reference {{ inputs }}';
  }
  // AFTER
  return instructions.includes('{{ inputs }}') || instructions.includes('{{ outputs }}')
    ? null
    : 'AFTER-stage instructions must reference {{ inputs }} or {{ outputs }}';
};
