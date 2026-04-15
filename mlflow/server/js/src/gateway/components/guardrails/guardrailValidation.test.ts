import { describe, test, expect } from '@jest/globals';
import { STAGE_HINTS, validateStageInstructions } from './guardrailValidation';

describe('STAGE_HINTS', () => {
  test('has entries for BEFORE and AFTER', () => {
    expect(STAGE_HINTS.BEFORE).toContain('{{ inputs }}');
    expect(STAGE_HINTS.AFTER).toContain('{{ inputs }}');
    expect(STAGE_HINTS.AFTER).toContain('{{ outputs }}');
  });
});

describe('validateStageInstructions', () => {
  // ─── Empty / whitespace ────────────────────────────────────────────────────

  test('returns null for empty string', () => {
    expect(validateStageInstructions('', 'BEFORE')).toBeNull();
    expect(validateStageInstructions('', 'AFTER')).toBeNull();
  });

  test('returns null for whitespace-only string', () => {
    expect(validateStageInstructions('   ', 'BEFORE')).toBeNull();
    expect(validateStageInstructions('\n\t', 'AFTER')).toBeNull();
  });

  // ─── BEFORE stage ─────────────────────────────────────────────────────────

  test('BEFORE: returns null when {{ inputs }} is present', () => {
    expect(validateStageInstructions('Is {{ inputs }} free of profanity?', 'BEFORE')).toBeNull();
  });

  test('BEFORE: returns null when both {{ inputs }} and {{ outputs }} are present (outputs error takes priority)', () => {
    // {{ outputs }} in BEFORE triggers the outputs-unavailable error regardless
    expect(validateStageInstructions('{{ inputs }} and {{ outputs }}', 'BEFORE')).toBe(
      '{{ outputs }} is not available in BEFORE stage — the LLM has not run yet',
    );
  });

  test('BEFORE: returns error when {{ inputs }} is missing', () => {
    expect(validateStageInstructions('Is this safe?', 'BEFORE')).toBe(
      'BEFORE-stage instructions must reference {{ inputs }}',
    );
  });

  test('BEFORE: returns outputs-unavailable error when {{ outputs }} is referenced', () => {
    expect(validateStageInstructions('Is {{ outputs }} appropriate?', 'BEFORE')).toBe(
      '{{ outputs }} is not available in BEFORE stage — the LLM has not run yet',
    );
  });

  // ─── AFTER stage ──────────────────────────────────────────────────────────

  test('AFTER: returns null when {{ inputs }} is present', () => {
    expect(validateStageInstructions('Does this answer {{ inputs }}?', 'AFTER')).toBeNull();
  });

  test('AFTER: returns null when {{ outputs }} is present', () => {
    expect(validateStageInstructions('Is {{ outputs }} safe?', 'AFTER')).toBeNull();
  });

  test('AFTER: returns null when both {{ inputs }} and {{ outputs }} are present', () => {
    expect(validateStageInstructions('Does {{ outputs }} correctly answer {{ inputs }}?', 'AFTER')).toBeNull();
  });

  test('AFTER: returns error when neither variable is referenced', () => {
    expect(validateStageInstructions('Is this safe?', 'AFTER')).toBe(
      'AFTER-stage instructions must reference {{ inputs }} or {{ outputs }}',
    );
  });
});
