import type { TSESTree } from '@typescript-eslint/utils';
import { ASTUtils } from '@typescript-eslint/utils';
import type { RuleContext } from '@typescript-eslint/utils/ts-eslint';

/**
 * Returns true if the given node is scoped to the top level of the module, or false if it's inside
 * a nested scope such as a function or class definition
 */
export function isAtModuleScope(context: RuleContext<string, unknown[]>, node: TSESTree.Node): boolean {
  const scope = ASTUtils.getInnermostScope(context.sourceCode.getScope(node), node);

  // We may have to accept `scope.type === 'global'` here too for a future use case
  return scope.type === 'module';
}
