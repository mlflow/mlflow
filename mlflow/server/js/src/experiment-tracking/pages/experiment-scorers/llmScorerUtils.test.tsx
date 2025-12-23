import { describe, it, expect } from '@jest/globals';
import { validateInstructions } from './llmScorerUtils';

describe('validateInstructions', () => {
  describe('Golden Path - Successful Operations', () => {
    it('should return true for valid instructions with {{ inputs }} variable', () => {
      const result = validateInstructions('Evaluate if the response in {{ inputs }} is correct');

      expect(result).toBe(true);
    });

    it('should return true for valid instructions with {{ outputs }} variable', () => {
      const result = validateInstructions('Check if {{ outputs }} matches the expected format');

      expect(result).toBe(true);
    });

    it('should return true for valid instructions with {{ expectations }} variable', () => {
      const result = validateInstructions('Verify that {{ expectations }} are met');

      expect(result).toBe(true);
    });

    it('should return true for valid instructions with {{ trace }} variable', () => {
      const result = validateInstructions('Analyze the full {{ trace }} for errors');

      expect(result).toBe(true);
    });

    it('should return true for valid instructions with {{ conversation }} variable alone', () => {
      const result = validateInstructions('Evaluate the {{ conversation }}');

      expect(result).toBe(true);
    });

    it('should return true for valid instructions with {{ conversation }} and {{ expectations }}', () => {
      const result = validateInstructions('Check {{ conversation }} against {{ expectations }}');

      expect(result).toBe(true);
    });

    it('should return true for valid instructions with multiple allowed variables', () => {
      const result = validateInstructions(
        'Compare {{ inputs }} with {{ outputs }} and check against {{ expectations }}',
      );

      expect(result).toBe(true);
    });

    it('should accept variables without spaces ({{inputs}})', () => {
      const result = validateInstructions('Check {{inputs}} and {{outputs}}');

      expect(result).toBe(true);
    });

    it('should accept variables with irregular spacing ({{ inputs}} or {{inputs }})', () => {
      const result = validateInstructions('Check {{ inputs}} and {{outputs }} for validity');

      expect(result).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should return true for empty string', () => {
      const result = validateInstructions('');

      expect(result).toBe(true);
    });

    it('should return true for undefined', () => {
      const result = validateInstructions(undefined);

      expect(result).toBe(true);
    });

    it('should return true for whitespace-only string', () => {
      const result = validateInstructions('   \n\t   ');

      expect(result).toBe(true);
    });

    it('should return true for instructions with same variable used multiple times', () => {
      const result = validateInstructions('Compare {{ inputs }} at the start with {{ inputs }} at the end');

      expect(result).toBe(true);
    });

    it('should handle instructions with braces that are not template variables', () => {
      const result = validateInstructions('Use {bracket} notation and {{ inputs }} variable');

      expect(result).toBe(true);
    });

    it('should handle malformed variables with extra braces', () => {
      const result = validateInstructions('Check {{{ inputs }}} with extra braces');

      expect(result).toBe(true);
    });

    it('should require template variables even when malformed braces are present', () => {
      const result = validateInstructions('Use {{ inputs without closing or closing only }}');

      expect(result).toBe(
        'Must contain at least one variable: {{ inputs }}, {{ outputs }}, {{ expectations }}, {{ trace }}, or {{ conversation }}',
      );
    });
  });

  describe('Error Conditions', () => {
    it('should return error message for invalid variable name', () => {
      const result = validateInstructions('Check {{ invalid }} variable');

      expect(result).toBe(
        'Invalid variable: {{ invalid }}. Only {{ inputs }}, {{ outputs }}, {{ expectations }}, {{ trace }}, {{ conversation }} are allowed',
      );
    });

    it('should return error message for multiple invalid variables', () => {
      const result = validateInstructions('Check {{ foo }} and {{ bar }} variables');

      expect(typeof result).toBe('string');
      expect(result).toContain('Invalid variable:');
    });

    it('should return error message when no template variables are present', () => {
      const result = validateInstructions('This instruction has no template variables');

      expect(result).toBe(
        'Must contain at least one variable: {{ inputs }}, {{ outputs }}, {{ expectations }}, {{ trace }}, or {{ conversation }}',
      );
    });

    it('should detect invalid variable even when valid variables are present', () => {
      const result = validateInstructions('Check {{ inputs }} and {{ invalid_var }} together');

      expect(result).toContain('Invalid variable: {{ invalid_var }}');
    });

    it('should validate case-sensitive variable names', () => {
      const result = validateInstructions('Check {{ Inputs }} with capital I');

      expect(result).toContain('Invalid variable: {{ Inputs }}');
    });

    it('should reject variables with special characters', () => {
      const result = validateInstructions('Check {{ input-value }} with dash');

      expect(result).toBe(
        'Must contain at least one variable: {{ inputs }}, {{ outputs }}, {{ expectations }}, {{ trace }}, or {{ conversation }}',
      );
    });

    it('should reject variables with numbers mixed in', () => {
      const result = validateInstructions('Check {{ input123 }} with numbers');

      expect(result).toContain('Invalid variable: {{ input123 }}');
    });
  });

  describe('Conversation Variable Restrictions', () => {
    it('should reject {{ conversation }} when used with {{ inputs }}', () => {
      const result = validateInstructions('Check {{ conversation }} with {{ inputs }}');

      expect(result).toBe(
        '{{ conversation }} can only be used with {{ expectations }}. Remove {{ inputs }} to use {{ conversation }}',
      );
    });

    it('should reject {{ conversation }} when used with {{ outputs }}', () => {
      const result = validateInstructions('Check {{ conversation }} with {{ outputs }}');

      expect(result).toBe(
        '{{ conversation }} can only be used with {{ expectations }}. Remove {{ outputs }} to use {{ conversation }}',
      );
    });

    it('should reject {{ conversation }} when used with {{ trace }}', () => {
      const result = validateInstructions('Check {{ conversation }} with {{ trace }}');

      expect(result).toBe(
        '{{ conversation }} can only be used with {{ expectations }}. Remove {{ trace }} to use {{ conversation }}',
      );
    });

    it('should reject {{ conversation }} when used with multiple disallowed variables', () => {
      const result = validateInstructions('Check {{ conversation }} with {{ inputs }} and {{ outputs }}');

      expect(result).toContain('{{ conversation }} can only be used with {{ expectations }}');
      expect(result).toContain('inputs');
      expect(result).toContain('outputs');
    });

    it('should reject {{ conversation }} when used with all other variables', () => {
      const result = validateInstructions(
        'Check {{ conversation }} with {{ inputs }}, {{ outputs }}, {{ expectations }}, and {{ trace }}',
      );

      expect(result).toContain('{{ conversation }} can only be used with {{ expectations }}');
    });
  });
});
