import type { TSESTree } from '@typescript-eslint/utils';
import { createRuleWithoutOptions } from '../utils/createRule';
import type { RuleContext } from '@typescript-eslint/utils/ts-eslint';

type MessageIds = 'passiveButtonLabel';

const FORBIDDEN_LABELS = new Set(['Confirm', 'Ok', 'Okay', 'Yes', 'No']);

/**
 * Helper to check if a node is inside an imported Modal/DangerModal component
 */
function findModalAncestor(
  context: Readonly<RuleContext<'passiveButtonLabel', []>>,
  node: TSESTree.Node,
  modalNames: Set<string>,
): TSESTree.Node | undefined {
  const sourceCode = context.sourceCode;
  return sourceCode
    .getAncestors(node)
    .find(
      (a: TSESTree.Node) =>
        a.type === 'JSXOpeningElement' && a.name.type === 'JSXIdentifier' && modalNames.has(a.name.name),
    );
}

export default createRuleWithoutOptions<MessageIds>({
  name: 'no-passive-modal-button-labels',
  meta: {
    type: 'problem',
    docs: {
      description: 'Avoid passive/vague button labels in Modal components.',
    },
    messages: {
      passiveButtonLabel:
        'Avoid vague button labels such as "Confirm" or "Okay". Use action-oriented verbs that match the verb in the modal header.',
    },
    fixable: undefined,
  },
  create(context) {
    const modalNames = new Set<string>();

    /**
     * Check if a literal node has a forbidden label and is inside a Modal
     */
    function checkLiteral(node: TSESTree.Literal) {
      if (!FORBIDDEN_LABELS.has(node.value as string)) return;
      if (!findModalAncestor(context, node, modalNames)) return;

      // Find okText or cancelText attribute node to report on using sourceCode.getAncestors(node)
      const reportNode = context.sourceCode
        .getAncestors(node)
        .find(
          (a: TSESTree.Node) => a.type === 'JSXAttribute' && (a.name.name === 'okText' || a.name.name === 'cancelText'),
        );

      context.report({
        node: reportNode ?? node,
        messageId: 'passiveButtonLabel',
        data: { label: node.value },
      });
    }

    return {
      ImportDeclaration(node: TSESTree.ImportDeclaration) {
        if (node.source.value === '@databricks/design-system') {
          for (const spec of node.specifiers) {
            if (spec.type === 'ImportSpecifier') {
              const importedName = spec.imported.type === 'Identifier' ? spec.imported.name : spec.imported.value;
              if (importedName === 'Modal' || importedName === 'DangerModal') {
                modalNames.add(spec.local.name);
              }
            }
          }
        }
      },

      // Direct literal: okText="Confirm"
      'JSXAttribute[name.name=/^(okText|cancelText)$/] Literal': checkLiteral,
    };
  },
});
