import { createRuleWithoutOptions } from '../utils/createRule';

import { getReferenceTracker, findFunctionCalls } from '../utils/trackReferences';
import { isAtModuleScope } from '../utils/isAtModuleScope';

type MessageIds = 'reactLazyNotAtModuleRoot';

export default createRuleWithoutOptions<MessageIds>({
  name: 'react-lazy-at-top-level',
  meta: {
    type: 'problem',
    docs: {
      description: 'Require React.lazy() calls to be at the module root level.',
    },
    messages: {
      reactLazyNotAtModuleRoot:
        'React.lazy() components must be defined at the top level of the module. Move this call outside of any functions, classes, or other nested scopes to ensure proper code splitting and lazy loading.',
    },
    fixable: undefined,
  },
  create(context) {
    return {
      'Program:exit'() {
        const tracker = getReferenceTracker(context);

        const functionCalls = findFunctionCalls(tracker, { module: 'react', functionName: 'lazy' });
        for (const node of functionCalls) {
          if (!isAtModuleScope(context, node)) {
            context.report({
              node: node.type === 'CallExpression' ? node.callee : node,
              messageId: 'reactLazyNotAtModuleRoot',
            });
          }
        }
      },
    };
  },
});
