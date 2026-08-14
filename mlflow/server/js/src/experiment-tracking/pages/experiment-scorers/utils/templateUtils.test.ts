import { describe, it, expect } from '@jest/globals';
import { hasTemplateVariable, TEMPLATE_VARIABLE_PATTERN } from './templateUtils';

describe('hasTemplateVariable', () => {
  it('should return true when template contains the variable', () => {
    expect(hasTemplateVariable('{{ expectations }}', 'expectations')).toBe(true);
  });

  it('should return true with variable whitespace inside braces', () => {
    expect(hasTemplateVariable('{{expectations}}', 'expectations')).toBe(true);
    expect(hasTemplateVariable('{{   expectations   }}', 'expectations')).toBe(true);
  });

  it('should return true when variable is embedded in longer text', () => {
    expect(hasTemplateVariable('Compare {{ outputs }} against {{ expectations }}', 'expectations')).toBe(true);
  });

  it('should return false when variable is not present', () => {
    expect(hasTemplateVariable('{{ outputs }}', 'expectations')).toBe(false);
  });

  it('should return false for undefined or empty template', () => {
    expect(hasTemplateVariable(undefined, 'expectations')).toBe(false);
    expect(hasTemplateVariable('', 'expectations')).toBe(false);
  });

  it('should not match partial variable names', () => {
    expect(hasTemplateVariable('{{ expected }}', 'expectations')).toBe(false);
  });
});

describe('TEMPLATE_VARIABLE_PATTERN', () => {
  it('should match template variables in a string', () => {
    const text = 'Compare {{ outputs }} against {{ expectations }}';
    const matches = text.match(TEMPLATE_VARIABLE_PATTERN);
    expect(matches).toEqual(['{{ outputs }}', '{{ expectations }}']);
  });

  it('should match variables with varying whitespace', () => {
    const text = '{{outputs}} and {{  inputs  }}';
    const matches = text.match(TEMPLATE_VARIABLE_PATTERN);
    expect(matches).toEqual(['{{outputs}}', '{{  inputs  }}']);
  });
});
