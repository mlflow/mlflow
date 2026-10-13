import type { TSESTree } from '@typescript-eslint/utils';
import { createRuleWithoutOptions } from '../utils/createRule';

type MessageIds = 'disallowWindowTop';

export default createRuleWithoutOptions<MessageIds>({
  name: 'no-window-top',
  meta: {
    type: 'problem',
    docs: {
      description: 'Disallow use of window.top.',
    },
    messages: {
      disallowWindowTop:
        'Do not use window.top. Please import `getWindowTop()` from `@databricks/web-shared/utils` instead.',
    },
    fixable: undefined,
  },
  create(context) {
    return {
      MemberExpression(node: TSESTree.MemberExpression) {
        if (
          node.object.type === 'Identifier' &&
          node.object.name === 'window' &&
          node.property.type === 'Identifier' &&
          node.property.name === 'top'
        ) {
          context.report({ node, messageId: 'disallowWindowTop' });
        }
      },
    };
  },
});
