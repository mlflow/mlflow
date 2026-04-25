import type { GuardrailStage } from '../../types';

export const STAGE_HINTS: Record<GuardrailStage, string> = {
  BEFORE:
    'Receives {{ inputs }} (the incoming request). Answer yes to pass, no to block/sanitize.\nExample: "Is {{ inputs }} free of profanity? Answer yes if it is safe, no if it contains profanity."',
  AFTER:
    'Receives {{ inputs }} (the request) and {{ outputs }} (the response). Answer yes to pass, no to block/sanitize.\nExample: "Does {{ outputs }} correctly answer {{ inputs }}? Answer yes if it does, no if it is off-topic or incorrect."',
};

/** Returns an error message if instructions are incompatible with the stage, or null if valid. */
export const validateStageInstructions = (instructions: string, stage: GuardrailStage): string | null => {
  if (!instructions.trim()) return null;
  if (stage === 'BEFORE') {
    if (instructions.includes('{{ outputs }}')) {
      return '{{ outputs }} is not available in Pre-LLM Guardrails — the LLM has not run yet';
    }
    return instructions.includes('{{ inputs }}') ? null : 'Pre-LLM Guardrails instructions must reference {{ inputs }}';
  }
  // AFTER
  return instructions.includes('{{ inputs }}') || instructions.includes('{{ outputs }}')
    ? null
    : 'Post-LLM Guardrails instructions must reference {{ inputs }} or {{ outputs }}';
};
