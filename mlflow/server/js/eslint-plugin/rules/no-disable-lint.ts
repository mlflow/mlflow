import { createRuleWithoutOptions } from '../utils/createRule';

type MessageIds = 'disableNotAllowed';

/**
 * Rules that must not be disabled via eslint-disable comments.
 * Use the full rule name including the @databricks/ prefix.
 */
const PROTECTED_RULES = new Set(['@databricks/no-dynamic-property-value']);

export default createRuleWithoutOptions<MessageIds>({
  name: 'no-disable-lint',
  meta: {
    type: 'problem',
    docs: {
      description: 'Prevents eslint-disable comments from suppressing certain protected lint rules.',
    },
    messages: {
      disableNotAllowed:
        'Disabling "{{ruleName}}" is not allowed. Fix the underlying issue instead of suppressing the lint error.',
    },
    fixable: undefined,
  },
  create(context) {
    const sourceCode = context.sourceCode;

    return {
      Program() {
        // Get all comments in the file
        const comments = sourceCode.getAllComments();

        for (const comment of comments) {
          // Only process comments that contain eslint-disable
          const commentValue = comment.value.trim();
          if (!commentValue.includes('eslint-disable')) {
            continue;
          }

          // Extract rule names from the disable comment.
          // Handles: eslint-disable ruleName, eslint-disable-next-line ruleName, eslint-disable-line ruleName
          // Also handles multiple rules: eslint-disable rule1, rule2
          const match = commentValue.match(/eslint-disable(?:-next-line|-line)?\s+(.+?)(?:\s*--|$)/);
          if (!match) {
            continue;
          }

          const rulesPart = match[1];
          const rules = rulesPart
            .split(',')
            .map((r) => r.trim())
            .filter(Boolean);

          for (const ruleName of rules) {
            if (PROTECTED_RULES.has(ruleName)) {
              context.report({
                loc: comment.loc,
                messageId: 'disableNotAllowed',
                data: { ruleName },
              });
            }
          }
        }
      },
    };
  },
});
