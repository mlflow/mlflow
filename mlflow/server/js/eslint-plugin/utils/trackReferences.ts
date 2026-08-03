import type { TSESTree } from '@typescript-eslint/utils';
import { ASTUtils } from '@typescript-eslint/utils';
import type { RuleContext } from '@typescript-eslint/utils/ts-eslint';

type ReferenceTrackerInstance = InstanceType<typeof ASTUtils.ReferenceTracker>;
type NodeIterator = IterableIterator<TSESTree.Node>;

export type FunctionImport = {
  /** Path of the module imported from. e.g. 'react' or './components/MyComponent' */
  module: string;

  /**
   * Name of the specific function imported from the module. e.g. 'useState' from React. Or, if the function
   * being called is the default export of the module, use DEFAULT_EXPORT
   */
  functionName: string;
};

export function getReferenceTracker(context: RuleContext<string, unknown[]>): ReferenceTrackerInstance {
  return new ASTUtils.ReferenceTracker(context.sourceCode.getScope(context.sourceCode.ast));
}

/**
 * For the given named import of a function, returns an iterator of all the function call nodes that call that function,
 * regardless of they way they were imported or whether they were renamed.
 *
 * @param tracker A ReferenceTracker instance
 * @param importConfig The name of the module and import to track
 * @returns An iterator of ESTree Nodes
 */
export function* findFunctionCalls(tracker: ReferenceTrackerInstance, importConfig: FunctionImport): NodeIterator {
  const traceMap = {
    [importConfig.module]: {
      // Specifies that we're looking at an ES Module (all our TypeScript files are ES modules)
      [ASTUtils.ReferenceTracker.ESM]: true,

      // Named imports: e.g. `import { useState } from 'react'`
      [importConfig.functionName]: {
        [ASTUtils.ReferenceTracker.CALL]: true,
      },

      // Default imports: `import React from 'react'`, or namespace imports: `import * as React from 'react'`
      default: {
        [importConfig.functionName]: {
          [ASTUtils.ReferenceTracker.CALL]: true,
        },
      },
    },
  } as const;

  for (const { node } of tracker.iterateEsmReferences(traceMap)) {
    yield node;
  }
}

/**
 * For the default import of a given module, returns an iterator of all the nodes that call that value.
 *
 * @param tracker A ReferenceTracker instance
 * @param importConfig The name of the imported module
 * @returns An iterator of ESTree Nodes
 */
export function* findFunctionCallsForDefaultExport(
  tracker: ReferenceTrackerInstance,
  importConfig: Omit<FunctionImport, 'functionName'>,
): NodeIterator {
  const traceMap = {
    [importConfig.module]: {
      // Specifies that we're looking at an ES Module (all our TypeScript files are ES modules)
      [ASTUtils.ReferenceTracker.ESM]: true,

      // Default imports: `import React from 'react'`, or namespace imports: `import * as React from 'react'`
      default: {
        [ASTUtils.ReferenceTracker.CALL]: true,
      },
    },
  } as const;

  for (const { node } of tracker.iterateEsmReferences(traceMap)) {
    yield node;
  }
}
