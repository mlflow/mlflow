// Pattern for template variables like {{ inputs }}, {{ outputs }}, etc.
export const TEMPLATE_VARIABLE_PATTERN = /(\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\})/g;

/**
 * Check if a template string contains a specific variable (handles variable whitespace).
 * e.g., hasTemplateVariable('{{ expectations }}', 'expectations') returns true
 *       hasTemplateVariable('{{   expectations   }}', 'expectations') returns true
 */
export const hasTemplateVariable = (template: string | undefined, variableName: string): boolean => {
  if (!template) return false;
  const pattern = new RegExp(`\\{\\{\\s*${variableName}\\s*\\}\\}`);
  return pattern.test(template);
};
