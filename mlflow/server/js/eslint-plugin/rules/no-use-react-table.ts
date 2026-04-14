import type { TSESTree } from '@typescript-eslint/utils';
import { createRule } from '../utils/createRule';

type MessageIds = 'disallowUseReactTable' | 'disallowWrapper';
type Options = [{ requireWrapper: boolean }];

export default createRule<Options, MessageIds>({
  name: 'no-use-react-table',
  meta: {
    type: 'problem',
    docs: {
      description: 'Disallow direct import of useReactTable from @tanstack/react-table.',
    },
    messages: {
      disallowUseReactTable:
        'Direct import of useReactTable from @tanstack/react-table is not allowed. Please import one of the useReactTable wrappers from @databricks/web-shared/react-table instead.',
      disallowWrapper:
        "Don't use the useReactTable wrapper functions in test files; import directly from @tanstack/react-table.",
    },
    schema: [
      {
        type: 'object',
        properties: {
          requireWrapper: { type: 'boolean' },
        },
        additionalProperties: false,
        required: ['requireWrapper'],
      },
    ],
    fixable: undefined,
  },
  defaultOptions: [{ requireWrapper: false }],
  create(context) {
    const [{ requireWrapper }] = context.options;

    if (requireWrapper) {
      return {
        'ImportDeclaration[source.value="@tanstack/react-table"] ImportSpecifier[imported.name="useReactTable"]'(
          node: TSESTree.ImportSpecifier,
        ) {
          context.report({ node, messageId: 'disallowUseReactTable' });
        },
      };
    } else {
      return {
        'ImportDeclaration[source.value="@databricks/web-shared/react-table"] ImportSpecifier[imported.name="useReactTable_unverifiedWithReact18"]'(
          node: TSESTree.ImportSpecifier,
        ) {
          context.report({ node, messageId: 'disallowWrapper' });
        },
      };
    }
  },
});
