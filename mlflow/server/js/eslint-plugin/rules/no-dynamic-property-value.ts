/**
 * ESLint rule: no-dynamic-property-value
 *
 * Prevents PII (Personally Identifiable Information) from being logged by enforcing that
 * certain properties in logging/observability functions must be statically determinable
 * at code-review time.
 *
 * ## Problem Statement
 *
 * Logging user data (usernames, emails, IDs from requests, etc.) creates privacy and
 * compliance risks. This rule ensures that event IDs, entity IDs, and component IDs used
 * in logging are static strings that can be reviewed during code review, not runtime
 * values that might contain user data.
 *
 * **Allowed (Static) Values:**
 * - String literals: `'user_login'`
 * - Const variables: `const id = 'login';`
 * - Enum values: `MyEnum.VALUE`
 * - Allowed parameters: `componentId` (from React components, not hooks)
 * - Parameters with enum types: `source: MyEnumType`
 * - TypeScript `satisfies` with allowed type names: `source satisfies MyEnumType`
 * - Template literals with static parts: `` `form-${componentId}` ``
 * - Ternary/logical expressions with static operands
 *
 * **Disallowed (Dynamic) Values:**
 * - Function calls: `generateId()`
 * - Let/var variables (mutable)
 * - Non-allowed parameters or renamed parameters: `{ id: componentId }`
 * - Hook parameters: `function useHook(componentId) { ... }`
 * - Object property access: `props.userId`
 *
 * ## Key Features
 *
 * **Import Alias Resolution:** Tracks `import { MyEnum as M }` and resolves correctly
 *
 * **Nested Scopes:** Allows parameter passing through nested functions
 *
 * **Type-Safe `satisfies`:** Allows `satisfies` expressions with enum types for compile-time safety
 */

import { type TSESTree, AST_NODE_TYPES } from '@typescript-eslint/utils';
import { createRuleWithoutOptions } from '../utils/createRule';

type MessageIds = 'dynamicPropertyValue' | 'propertyValueFromVariable';

/**
 * Configuration for allowed parameter patterns
 */
type AllowedParamConfig =
  | {
      type: 'simple';
      name: string; // Just check the parameter name directly
    }
  | {
      type: 'object-property';
      argPosition: number; // Which argument position (0-based)
      propertyPath: string[]; // Path to the allowed property (e.g., ['componentId'])
    };

/**
 * Configuration for checking a specific property path
 */
interface PropertyCheckConfig {
  propertyPath: string[];
  allowedParams: AllowedParamConfig[];
  /** Namespaces that can appear anywhere in the enum chain or be after `satisfies`. */
  allowedEnumPrefixesAndTypes: string[];
  /** If true, allows undefined as a valid value */
  allowUndefined?: boolean;
}

/**
 * Configuration for a target function type
 */
interface FunctionConfig {
  functionNames: string[];
  className?: string; // For class-based methods like MyUtils.method()
  getArgumentIndex: (node: TSESTree.CallExpression) => number;
  propertyChecks: PropertyCheckConfig[];
  // If true, requires inline ObjectExpression
  requireInlineObject?: boolean;
}

/**
 * Configuration for JSX attribute checking
 */
interface JSXAttributeConfig {
  attributeName: string;
  componentNameList?: string[];
  propertyCheck: PropertyCheckConfig;
}

/**
 * Configuration map for all target functions.
 *
 * To add a new function to check, add a new entry to this array with:
 * - functionNames: Array of function names to match
 * - className: (optional) Class name for method calls like ClassName.method()
 * - getArgumentIndex: Function that returns which argument to check (0-based)
 * - propertyChecks: Array of property paths and their allowed values
 * - requireInlineObject: (optional) If true, the argument must be an inline object literal
 */
const FUNCTION_CONFIGS: FunctionConfig[] = [
  // Example: add your logging/observability function configurations here
  {
    functionNames: ['logTelemetryEvent'],
    getArgumentIndex: () => 0,
    requireInlineObject: true,
    propertyChecks: [
      {
        propertyPath: ['componentId'],
        allowedParams: [{ type: 'object-property', argPosition: 0, propertyPath: ['componentId'] }],
        allowedEnumPrefixesAndTypes: [],
      },
    ],
  },
  {
    functionNames: ['useDesignSystemEventComponentCallbacks'],
    getArgumentIndex: () => 0,
    requireInlineObject: true,
    propertyChecks: [
      {
        propertyPath: ['componentId'],
        allowedParams: [{ type: 'object-property', argPosition: 0, propertyPath: ['componentId'] }],
        allowedEnumPrefixesAndTypes: [],
      },
    ],
  },
];

/**
 * Configuration for JSX attributes to check.
 *
 * To add a new JSX attribute to check, add a new entry with:
 * - attributeName: The JSX attribute name to validate
 * - componentNameList: (optional) Only check on these components
 * - propertyCheck: Validation configuration for the attribute value
 */
const JSX_ATTRIBUTE_CONFIGS: JSXAttributeConfig[] = [
  {
    attributeName: 'componentId',
    propertyCheck: {
      propertyPath: [],
      allowedParams: [{ type: 'object-property', argPosition: 0, propertyPath: ['componentId'] }],
      allowedEnumPrefixesAndTypes: [],
    },
  },
  {
    attributeName: 'description',
    componentNameList: ['LoadingState'],
    propertyCheck: {
      propertyPath: [],
      allowedParams: [{ type: 'object-property', argPosition: 0, propertyPath: ['componentId'] }],
      allowedEnumPrefixesAndTypes: ['LoadingDescription'],
      allowUndefined: true,
    },
  },
  {
    attributeName: 'loadingDescription',
    componentNameList: [
      'Button',
      'Card',
      'LegacySelect',
      'LegacySkeleton',
      'GenericSkeleton',
      'ParagraphSkeleton',
      'TableSkeletonRows',
      'TitleSkeleton',
      'Spinner',
      'SpinnerSuspense',
    ],
    propertyCheck: {
      propertyPath: [],
      allowedParams: [{ type: 'object-property', argPosition: 0, propertyPath: ['componentId'] }],
      allowedEnumPrefixesAndTypes: ['LoadingDescription'],
      allowUndefined: true,
    },
  },
];

/**
 * Check if a Literal node is static (only string literals are allowed).
 *
 * @param node - The Literal AST node to check
 * @returns true if the literal is a string, false otherwise
 *
 * @example Valid string literals (returns true):
 * ```typescript
 * 'static-event-id'
 * "another-string"
 * `simple-template-string`
 * ```
 *
 * @example Invalid literals (returns false):
 * ```typescript
 * 123        // number
 * true       // boolean
 * null       // null
 * undefined  // undefined
 * ```
 */
function isStaticLiteral(node: TSESTree.Literal): boolean {
  return typeof node.value === 'string';
}

/**
 * Get the set of simple allowed parameter names from the configuration
 */
function getSimpleAllowedParamNames(propertyCheck: PropertyCheckConfig): Set<string> {
  const names = new Set<string>();
  for (const param of propertyCheck.allowedParams) {
    if (param.type === 'simple') {
      names.add(param.name);
    }
  }
  return names;
}

/**
 * Check if an identifier is a direct function parameter (not destructured).
 *
 * @param node - The Identifier AST node
 * @param paramName - The parameter name to check
 * @returns true if this is a direct parameter like function Foo(paramName)
 *
 * @example Direct parameter (returns true):
 * ```typescript
 * function Foo(componentId: string) { ... }
 * ```
 *
 * @example Destructured parameter (returns false):
 * ```typescript
 * function Foo({ componentId }: Props) { ... }
 * ```
 */
function isDirectFunctionParameter(node: TSESTree.Identifier, paramName: string): boolean {
  let current: TSESTree.Node | undefined = node;

  while (current) {
    if (
      current.type === AST_NODE_TYPES.FunctionDeclaration ||
      current.type === AST_NODE_TYPES.FunctionExpression ||
      current.type === AST_NODE_TYPES.ArrowFunctionExpression
    ) {
      for (const param of current.params) {
        if (param.type === AST_NODE_TYPES.Identifier && param.name === paramName) {
          return true;
        }
        if (
          param.type === AST_NODE_TYPES.AssignmentPattern &&
          param.left.type === AST_NODE_TYPES.Identifier &&
          param.left.name === paramName
        ) {
          return true;
        }
      }
    }
    current = current.parent;
  }

  return false;
}

/**
 * Check if an identifier is destructured from an object parameter at a specific position.
 *
 * @param node - The Identifier AST node
 * @param paramConfig - The object-property configuration
 * @param functionParams - Set of function parameter names
 * @returns true if this identifier is destructured from the correct parameter position
 *
 * @example Matches config { argPosition: 0, propertyPath: ['componentId'] }:
 * ```typescript
 * function Foo({ componentId }: Props) {  // componentId destructured from arg 0
 *   return <div componentId={componentId} />;  // returns true
 * }
 * ```
 *
 * @example Does NOT match:
 * ```typescript
 * function Foo({ barProps }: Props) {  // barProps at arg 0, not componentId
 *   return <div componentId={barProps.componentId} />;  // returns false
 * }
 * ```
 */
function isDestructuredFromObjectParameter(
  node: TSESTree.Identifier,
  paramConfig: Extract<AllowedParamConfig, { type: 'object-property' }>,
  functionParams: Set<string>,
): boolean {
  if (!functionParams.has(node.name)) {
    return false;
  }

  let current: TSESTree.Node | undefined = node;

  while (current) {
    if (
      current.type === AST_NODE_TYPES.FunctionDeclaration ||
      current.type === AST_NODE_TYPES.FunctionExpression ||
      current.type === AST_NODE_TYPES.ArrowFunctionExpression
    ) {
      const param = current.params[paramConfig.argPosition];
      if (!param) {
        current = current.parent;
        continue;
      }

      if (param.type === AST_NODE_TYPES.ObjectPattern) {
        const matchesPath = checkObjectPatternMatchesPath(param, paramConfig.propertyPath, node.name);
        if (matchesPath && !isParameterFromHook(node.name, node)) {
          return true;
        }
      }

      if (param.type === AST_NODE_TYPES.AssignmentPattern && param.left.type === AST_NODE_TYPES.ObjectPattern) {
        const matchesPath = checkObjectPatternMatchesPath(param.left, paramConfig.propertyPath, node.name);
        if (matchesPath && !isParameterFromHook(node.name, node)) {
          return true;
        }
      }
    }
    current = current.parent;
  }

  return false;
}

/**
 * Check if an ObjectPattern contains a property path that matches the identifier.
 *
 * @param pattern - The ObjectPattern to check
 * @param propertyPath - The expected property path (e.g., ['componentId'])
 * @param identifierName - The identifier name to match
 * @returns true if the pattern contains the property path matching the identifier
 */
function checkObjectPatternMatchesPath(
  pattern: TSESTree.ObjectPattern,
  propertyPath: string[],
  identifierName: string,
): boolean {
  // For now, we only support single-level paths like ['componentId']
  // Multi-level paths like ['nested', 'componentId'] would require nested ObjectPatterns
  if (propertyPath.length !== 1) {
    return false;
  }

  const expectedProp = propertyPath[0];

  for (const prop of pattern.properties) {
    if (prop.type === AST_NODE_TYPES.Property) {
      const propKey =
        prop.key.type === AST_NODE_TYPES.Identifier
          ? prop.key.name
          : prop.key.type === AST_NODE_TYPES.Literal
            ? String(prop.key.value)
            : null;

      const propValue =
        prop.value.type === AST_NODE_TYPES.Identifier
          ? prop.value.name
          : prop.value.type === AST_NODE_TYPES.AssignmentPattern && prop.value.left.type === AST_NODE_TYPES.Identifier
            ? prop.value.left.name
            : null;

      if (propKey === expectedProp && propValue === identifierName) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Check if an identifier is a renamed destructured function parameter where the source
 * property name is NOT in the allowed list.
 *
 * This catches cases like `function MyComponent({ id: componentId })` where the local
 * variable is named `componentId` (an allowed name) but the source property is `id`
 * (not allowed). We should reject this because the source property name matters.
 *
 * @param node - The Identifier AST node to check
 * @param propertyCheck - Configuration defining what parameters are allowed
 * @returns true if this is a renamed parameter with non-allowed source property
 *
 * @example Renamed parameter with non-allowed source (returns true):
 * ```typescript
 * function MyComponent({ id: componentId }: { id: string }) {
 *   // Source property is 'id' (NOT allowed), local name is 'componentId' (allowed)
 *   return <Component componentId={componentId} />; // Should be rejected
 * }
 * ```
 *
 * @example Normal allowed parameter (returns false):
 * ```typescript
 * function MyComponent({ componentId }: Props) {
 *   // Source property is 'componentId' (allowed), same as local name
 *   return <Component componentId={componentId} />; // OK
 * }
 * ```
 */
function isRenamedDestructuredParameter(node: TSESTree.Identifier, propertyCheck: PropertyCheckConfig): boolean {
  const allowedSimpleNames = getSimpleAllowedParamNames(propertyCheck);

  let current: TSESTree.Node | undefined = node.parent;

  while (current) {
    if (
      current.type === AST_NODE_TYPES.FunctionDeclaration ||
      current.type === AST_NODE_TYPES.FunctionExpression ||
      current.type === AST_NODE_TYPES.ArrowFunctionExpression
    ) {
      for (const param of current.params) {
        if (param.type === AST_NODE_TYPES.ObjectPattern) {
          for (const prop of param.properties) {
            if (prop.type === AST_NODE_TYPES.Property) {
              const propValue =
                prop.value.type === AST_NODE_TYPES.Identifier
                  ? prop.value.name
                  : prop.value.type === AST_NODE_TYPES.AssignmentPattern &&
                      prop.value.left.type === AST_NODE_TYPES.Identifier
                    ? prop.value.left.name
                    : null;

              if (propValue === node.name) {
                const propKey =
                  prop.key.type === AST_NODE_TYPES.Identifier
                    ? prop.key.name
                    : prop.key.type === AST_NODE_TYPES.Literal
                      ? String(prop.key.value)
                      : null;

                if (propKey && propKey !== propValue && !allowedSimpleNames.has(propKey)) {
                  return true;
                }
              }
            }
          }
        } else if (
          param.type === AST_NODE_TYPES.AssignmentPattern &&
          param.left.type === AST_NODE_TYPES.ObjectPattern
        ) {
          for (const prop of param.left.properties) {
            if (prop.type === AST_NODE_TYPES.Property) {
              const propValue =
                prop.value.type === AST_NODE_TYPES.Identifier
                  ? prop.value.name
                  : prop.value.type === AST_NODE_TYPES.AssignmentPattern &&
                      prop.value.left.type === AST_NODE_TYPES.Identifier
                    ? prop.value.left.name
                    : null;

              if (propValue === node.name) {
                const propKey =
                  prop.key.type === AST_NODE_TYPES.Identifier
                    ? prop.key.name
                    : prop.key.type === AST_NODE_TYPES.Literal
                      ? String(prop.key.value)
                      : null;

                if (propKey && propKey !== propValue && !allowedSimpleNames.has(propKey)) {
                  return true;
                }
              }
            }
          }
        }
      }
    }

    current = current.parent;
  }

  return false;
}

/**
 * Check if an Identifier node is static.
 *
 * Handles multiple cases:
 * 1. Simple direct parameters (e.g., componentId as a direct param, not destructured)
 * 2. Object-property parameters (destructured from specific arg position)
 * 3. Destructured parameters from const variables (e.g., const { componentId } = props)
 * 4. Function parameters with allowed enum type annotations
 * 5. Const variables initialized with static values
 *
 * @param node - The Identifier AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 * @returns true if the identifier is static and safe, false otherwise
 *
 * @example Simple parameter (returns true):
 * ```typescript
 * function MyComponent(componentId: string) {
 *   return <Component componentId={componentId} />; // Direct parameter
 * }
 * ```
 *
 * @example Object-property destructured parameter (returns true):
 * ```typescript
 * function MyComponent({ componentId }: Props) {
 *   return <Component componentId={componentId} />; // Destructured from arg 0
 * }
 * ```
 *
 * @example Const destructured from param (returns true):
 * ```typescript
 * function MyComponent(props: Props) {
 *   const { componentId } = props;
 *   return <Component componentId={componentId} />; // componentId from destructuring
 * }
 * ```
 *
 * @example Parameter with enum type (returns true):
 * ```typescript
 * function MyWrapper({ source }: { source: EnvironmentConfigurationFormSource }) {
 *   return <Component componentId={source} />; // source is typed as allowed enum
 * }
 * ```
 *
 * @example Const variable with static value (returns true):
 * ```typescript
 * const eventId = 'user_login';
 * recordProto({ observability_log: { entity: { entity_id: eventId } } });
 * ```
 *
 * @example Invalid: let variable (returns false):
 * ```typescript
 * let eventId = 'user_login'; // Mutable, could be reassigned
 * recordProto({ observability_log: { entity: { entity_id: eventId } } });
 * ```
 *
 * @example Invalid: parameter from hook (returns false):
 * ```typescript
 * function useMyHook(componentId: string) {
 *   return <Component componentId={componentId} />; // Not allowed from hooks
 * }
 * ```
 *
 * @example Invalid: renamed parameter (returns false):
 * ```typescript
 * function MyComponent({ id: componentId }: { id: string }) {
 *   return <Component componentId={componentId} />; // Source property is 'id', not 'componentId'
 * }
 * ```
 *
 * @example Invalid: destructured from wrong property (returns false):
 * ```typescript
 * function MyComponent({ barProps }: Props) {
 *   return <Component componentId={barProps.componentId} />; // Wrong property path
 * }
 */
function isStaticIdentifier(
  node: TSESTree.Identifier,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  if (propertyCheck.allowUndefined && node.name === 'undefined') {
    return true;
  }

  const allowedParams = propertyCheck.allowedParams;

  for (const paramConfig of allowedParams) {
    if (paramConfig.type === 'simple' && paramConfig.name === node.name && functionParams.has(node.name)) {
      if (isDirectFunctionParameter(node, paramConfig.name)) {
        return !isParameterFromHook(node.name, node);
      }
    }
  }

  for (const paramConfig of allowedParams) {
    if (paramConfig.type === 'object-property' && functionParams.has(node.name)) {
      if (isDestructuredFromObjectParameter(node, paramConfig, functionParams)) {
        return true;
      }
    }
  }

  if (isDestructuredFromAllowedParam(node, propertyCheck, functionParams)) {
    return true;
  }

  if (functionParams.has(node.name)) {
    if (isParameterWithEnumType(node, propertyCheck, importAliases)) {
      return true;
    }
  }

  return isConstWithStaticValue(node, propertyCheck, importAliases);
}

/**
 * Check if a member expression matches an object-property parameter configuration.
 *
 * Validates that:
 * 1. The object parameter is at the correct argument position
 * 2. The property path matches the configured path
 *
 * @param node - The AST node being validated
 * @param propertyCheck - Configuration with allowed parameters
 * @param objectName - Name of the object being accessed (e.g., 'props')
 * @param propertyPath - Path of properties being accessed (e.g., ['componentId'])
 * @returns true if this matches an object-property configuration
 *
 * @example Valid:
 * ```typescript
 * // Config: { type: 'object-property', argPosition: 0, propertyPath: ['componentId'] }
 * function MyComponent(props: Props) {  // props is at position 0
 *   return <Component componentId={props.componentId} />; // Matches!
 * }
 * ```
 *
 * @example Invalid:
 * ```typescript
 * // Config: { type: 'object-property', argPosition: 0, propertyPath: ['componentId'] }
 * function MyComponent(props: Props) {
 *   return <Component componentId={props.barProps.componentId} />; // Path doesn't match
 * }
 * ```
 */
function isMatchingObjectPropertyParam(
  node: TSESTree.Node,
  propertyCheck: PropertyCheckConfig,
  objectName: string,
  propertyPath: string[],
): boolean {
  let current: TSESTree.Node | undefined = node;
  while (current) {
    if (
      current.type === AST_NODE_TYPES.FunctionDeclaration ||
      current.type === AST_NODE_TYPES.FunctionExpression ||
      current.type === AST_NODE_TYPES.ArrowFunctionExpression
    ) {
      for (const paramConfig of propertyCheck.allowedParams) {
        if (paramConfig.type === 'object-property') {
          const param = current.params[paramConfig.argPosition];
          if (!param) continue;

          const paramName = getParamNameAtPosition(param, objectName);
          if (paramName === objectName) {
            if (arraysEqual(paramConfig.propertyPath, propertyPath)) {
              return true;
            }
          }
        }
      }
    }
    current = current.parent;
  }

  return false;
}

/**
 * Get the parameter name if it matches the expected name
 *
 * Handles:
 * - Direct identifiers: function Foo(props)
 * - Assignment patterns: function Foo(props = {})
 * - Rest parameters in destructuring: function Foo({ ...props })
 */
function getParamNameAtPosition(param: TSESTree.Parameter, expectedName: string): string | null {
  // Direct identifier: function Foo(props)
  if (param.type === AST_NODE_TYPES.Identifier && param.name === expectedName) {
    return param.name;
  }

  // Assignment pattern: function Foo(props = {})
  if (param.type === AST_NODE_TYPES.AssignmentPattern && param.left.type === AST_NODE_TYPES.Identifier) {
    return param.left.name === expectedName ? param.left.name : null;
  }

  // Rest parameter in object destructuring: function Foo({ ...props })
  if (param.type === AST_NODE_TYPES.ObjectPattern) {
    for (const prop of param.properties) {
      if (
        prop.type === AST_NODE_TYPES.RestElement &&
        prop.argument.type === AST_NODE_TYPES.Identifier &&
        prop.argument.name === expectedName
      ) {
        return prop.argument.name;
      }
    }
  }

  // Assignment pattern wrapping object pattern: function Foo({ ...props } = {})
  if (param.type === AST_NODE_TYPES.AssignmentPattern && param.left.type === AST_NODE_TYPES.ObjectPattern) {
    for (const prop of param.left.properties) {
      if (
        prop.type === AST_NODE_TYPES.RestElement &&
        prop.argument.type === AST_NODE_TYPES.Identifier &&
        prop.argument.name === expectedName
      ) {
        return prop.argument.name;
      }
    }
  }

  return null;
}

/**
 * Check if two arrays are equal
 */
function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Check if a MemberExpression is static.
 */
function isStaticMemberExpression(
  node: TSESTree.MemberExpression,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  if (isEnumAccess(node, propertyCheck, importAliases)) {
    return true;
  }

  if (node.object.type === AST_NODE_TYPES.Identifier && node.property.type === AST_NODE_TYPES.Identifier) {
    const objectName = node.object.name;
    const propertyName = node.property.name;

    if (functionParams.has(objectName) && !isParameterFromHook(objectName, node)) {
      const matchingParam = isMatchingObjectPropertyParam(node, propertyCheck, objectName, [propertyName]);
      if (matchingParam) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Check if a TypeScript type annotation is safe (contains only allowed enums or string literals).
 *
 * Recursively validates type annotations used in `satisfies` expressions to ensure they
 * only reference allowed enum types or static string literals.
 *
 * @param typeNode - The TypeScript type AST node to check
 * @param propertyCheck - Configuration defining what enums are allowed
 * @param importAliases - Map of import aliases for resolving enum names
 * @returns true if the type is safe (only contains allowed types), false otherwise
 *
 * @example Valid types (returns true):
 * ```typescript
 * EnvironmentConfigurationFormSource     // Allowed type name (enum)
 * ReadOnlyCodeBlockActionId              // Allowed type name (string union)
 * 'specific-value'                       // String literal
 * `prefix-${AllowedEnum}`                // Template with allowed type
 * `${AllowedEnum1}-${AllowedEnum2}`      // Template with multiple allowed types
 * `nested-${`inner-${AllowedEnum}`}`     // Nested template with allowed type
 * 'option1' | 'option2'                  // Union of string literals
 * AllowedEnum1 | AllowedEnum2            // Union of allowed types
 * 'literal' | `template-${AllowedEnum}`  // Mixed union
 * ```
 *
 * @example Invalid types (returns false):
 * ```typescript
 * SomeOtherType                          // Non-allowed type
 * `prefix-${NonAllowedEnum}`             // Template with non-allowed enum
 * 123                                    // Non-string literal
 * AllowedEnum | string                   // Union with broad 'string' type
 * typeof MyEnum                          // Type query (not supported)
 * AllowedEnum & SomeInterface            // Intersection type (not supported)
 * ```
 */
function isStaticTypeAnnotation(
  typeNode: TSESTree.TypeNode,
  propertyCheck: PropertyCheckConfig,
  importAliases: Map<string, string>,
): boolean {
  // Handle TSTypeReference (enum types like ComponentType, or string union types like ReadOnlyCodeBlockActionId)
  if (typeNode.type === AST_NODE_TYPES.TSTypeReference) {
    const typeName = getTypeNameFromTypeAnnotation(typeNode, importAliases);
    return matchesAllowedEnumOrUnionNamespace(typeName, propertyCheck, importAliases);
  }

  // Handle TSLiteralType (string literal types like 'specific-value')
  if (typeNode.type === AST_NODE_TYPES.TSLiteralType) {
    // Only string literal types are safe
    return typeNode.literal.type === AST_NODE_TYPES.Literal && typeof typeNode.literal.value === 'string';
  }

  // Handle TSTemplateLiteralType (template literal types like `prefix-${EnumType}`)
  if (typeNode.type === AST_NODE_TYPES.TSTemplateLiteralType) {
    // Recursively check all type parameters in the template
    return typeNode.types.every((type) => isStaticTypeAnnotation(type, propertyCheck, importAliases));
  }

  // Handle TSUnionType (union types like 'a' | 'b' or EnumA | EnumB)
  if (typeNode.type === AST_NODE_TYPES.TSUnionType) {
    // Allow unions where ALL members are safe
    // This handles: 'a' | 'b', AllowedEnum1 | AllowedEnum2, etc.
    return typeNode.types.every((type) => isStaticTypeAnnotation(type, propertyCheck, importAliases));
  }

  // Other type nodes are not supported (intersection types, typeof, indexed access, etc.)
  return false;
}

/**
 * Check if a TSSatisfiesExpression is static.
 *
 * TypeScript's `satisfies` operator provides compile-time type checking without changing
 * the expression's type. If the type annotation matches an allowed enum, we consider it
 * safe because TypeScript guarantees type safety at compile time.
 *
 * @param node - The TSSatisfiesExpression AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 * @returns true if the satisfies expression is static and safe, false otherwise
 *
 * @example Satisfies with allowed enum type (returns true):
 * ```typescript
 * function MyWrapper({ source }: RandomTypeHereThatIsImportedElseWhere) {
 *   return <Component componentId={source satisfies EnvironmentConfigurationFormSource} />;
 * }
 * ```
 *
 * @example Satisfies with import alias (returns true):
 * ```typescript
 * import { EnvironmentConfigurationFormSource as ECFS } from '@databricks/web-shared/types';
 * function MyWrapper({ source }: RandomTypeHereThatIsImportedElseWhere) {
 *   return <Component componentId={source satisfies ECFS} />;
 * }
 * ```
 *
 * @example Satisfies in template literal (returns true):
 * ```typescript
 * function MyWrapper({ source }: { source: EnvironmentConfigurationFormSource }) {
 *   return <Component componentId={`prefix_${source satisfies EnvironmentConfigurationFormSource}_suffix`} />;
 * }
 * ```
 *
 * @example Satisfies with function call - allowed due to type safety (returns true):
 * ```typescript
 * function MyWrapper() {
 *   return <Component componentId={getSource() satisfies EnvironmentConfigurationFormSource} />;
 * }
 * ```
 *
 * @example Satisfies with string literal type (returns true):
 * ```typescript
 * function MyWrapper() {
 *   return <Component componentId={getValue() satisfies 'specific-value'} />;
 * }
 * ```
 *
 * @example Satisfies with template literal type (returns true):
 * ```typescript
 * function MyWrapper() {
 *   return <Component componentId={getValue() satisfies `prefix-${ComponentType}`} />;
 * }
 * ```
 *
 * @example Satisfies with nested template literal type (returns true):
 * ```typescript
 * function MyWrapper() {
 *   return <Component componentId={getValue() satisfies `outer-${`inner-${ComponentType}`}`} />;
 * }
 * ```
 *
 * @example Invalid: satisfies with non-allowed type (returns false):
 * ```typescript
 * function MyWrapper({ source }: { source: string }) {
 *   return <Component componentId={source satisfies SomeOtherType} />;
 * }
 * ```
 */
function isStaticSatisfiesExpression(
  node: TSESTree.TSSatisfiesExpression,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  // Check if the satisfies type annotation is safe using recursive validation
  if (node.typeAnnotation && isStaticTypeAnnotation(node.typeAnnotation, propertyCheck, importAliases)) {
    // TypeScript's 'satisfies' operator ensures the value matches the type at compile time,
    // so it's safe regardless of where the value comes from (function param, variable, function call, etc.)
    return true;
  }

  // If the type annotation is not safe, recursively check the expression itself
  return isStaticValue(node.expression, propertyCheck, functionParams, importAliases);
}

/**
 * Check if a TemplateLiteral is static (all interpolated expressions must be static).
 *
 * Template literals are allowed if every interpolated expression (the parts inside ${...})
 * is itself a static value. The static string parts are always safe.
 *
 * @param node - The TemplateLiteral AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 * @returns true if all expressions in the template are static, false otherwise
 *
 * @example Template with static expressions (returns true):
 * ```typescript
 * const eventType = 'login';
 * `user_${eventType}_click` // eventType is const, so it's static
 * ```
 *
 * @example Template with enum values (returns true):
 * ```typescript
 * `form_${EnvironmentConfigurationFormField.BASE_ENVIRONMENT}_input`
 * ```
 *
 * @example Template with allowed parameter (returns true):
 * ```typescript
 * function MyComponent({ componentId }: Props) {
 *   return <Component componentId={`button_${componentId}`} />;
 * }
 * ```
 *
 * @example Template with multiple static parts (returns true):
 * ```typescript
 * function MyWrapper({ source }: { source: EnvironmentConfigurationFormSource }) {
 *   return <Component componentId={`environmentForm-${source}-${EnvironmentConfigurationFormField.BASE_ENVIRONMENT}.input`} />;
 * }
 * ```
 *
 * @example Template with satisfies expression (returns true):
 * ```typescript
 * `environmentForm-${source satisfies EnvironmentConfigurationFormSource}-field`
 * ```
 *
 * @example Invalid: template with function call (returns false):
 * ```typescript
 * `user_${computeId()}_event` // computeId() is dynamic
 * ```
 *
 * @example Invalid: template with non-allowed variable (returns false):
 * ```typescript
 * let userId = getUser().id;
 * `user_${userId}_event` // userId could contain PII
 * ```
 */
function isStaticTemplateLiteral(
  node: TSESTree.TemplateLiteral,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  return node.expressions.every((expr) => isStaticValue(expr, propertyCheck, functionParams, importAliases));
}

/**
 * Check if a ConditionalExpression (ternary) is static (both branches must be static).
 *
 * Ternary expressions (condition ? consequent : alternate) are allowed if both the
 * consequent (true branch) and alternate (false branch) are static values.
 * The condition itself can be dynamic since it doesn't affect the logged value.
 *
 * @param node - The ConditionalExpression AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 * @returns true if both branches are static, false otherwise
 *
 * @example Ternary with static branches (returns true):
 * ```typescript
 * isError ? 'error_event' : 'success_event'
 * ```
 *
 * @example Ternary with enum values (returns true):
 * ```typescript
 * hasButton ? ComponentType.BUTTON : ComponentType.INPUT
 * ```
 *
 * @example Ternary with const variables (returns true):
 * ```typescript
 * const errorId = 'error_login';
 * const successId = 'success_login';
 * isError ? errorId : successId
 * ```
 *
 * @example Ternary with allowed parameter (returns true):
 * ```typescript
 * function MyComponent({ componentId }: Props) {
 *   return <Component componentId={hasPrefix ? `prefix_${componentId}` : componentId} />;
 * }
 * ```
 *
 * @example Invalid: ternary with dynamic branch (returns false):
 * ```typescript
 * isError ? generateErrorId() : 'success_event' // generateErrorId() is dynamic
 * ```
 *
 * @example Invalid: ternary with non-allowed variable (returns false):
 * ```typescript
 * let userId = getUser().id;
 * isError ? 'error' : userId // userId could contain PII
 * ```
 */
function isStaticConditionalExpression(
  node: TSESTree.ConditionalExpression,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  return (
    isStaticValue(node.consequent, propertyCheck, functionParams, importAliases) &&
    isStaticValue(node.alternate, propertyCheck, functionParams, importAliases)
  );
}

/**
 * Check if a LogicalExpression is static (both operands must be static).
 *
 * Logical expressions (&&, ||, ??) are allowed if both the left and right operands
 * are static values. This handles cases like fallback values or conditional expressions.
 *
 * @param node - The LogicalExpression AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 * @returns true if both operands are static, false otherwise
 *
 * @example Logical OR with fallback (returns true):
 * ```typescript
 * const defaultId = 'default_event';
 * eventId || defaultId // Both sides are static
 * ```
 *
 * @example Logical AND (returns true):
 * ```typescript
 * hasFeature && ComponentType.BUTTON
 * ```
 *
 * @example Nullish coalescing with static values (returns true):
 * ```typescript
 * const eventId = null;
 * eventId ?? 'default_event'
 * ```
 *
 * @example With allowed parameters (returns true):
 * ```typescript
 * function MyComponent({ componentId }: Props) {
 *   return <Component componentId={componentId || 'default_id'} />;
 * }
 * ```
 *
 * @example Invalid: logical expression with function call (returns false):
 * ```typescript
 * eventId || generateId() // generateId() is dynamic
 * ```
 *
 * @example Invalid: logical expression with non-allowed variable (returns false):
 * ```typescript
 * let userId = getUser().id;
 * userId || 'default' // userId could contain PII
 * ```
 */
function isStaticLogicalExpression(
  node: TSESTree.LogicalExpression,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  return (
    isStaticValue(node.left, propertyCheck, functionParams, importAliases) &&
    isStaticValue(node.right, propertyCheck, functionParams, importAliases)
  );
}

/**
 * Check if a BinaryExpression is static.
 *
 * Only the '+' operator is supported (for string concatenation). Both operands must be static.
 * Other operators are not allowed as they typically don't produce static strings.
 *
 * @param node - The BinaryExpression AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 * @returns true if the operator is '+' and both operands are static, false otherwise
 *
 * @example String concatenation with static values (returns true):
 * ```typescript
 * const prefix = 'user';
 * const suffix = 'event';
 * prefix + '_' + suffix // All parts are static
 * ```
 *
 * @example Concatenation with enum (returns true):
 * ```typescript
 * 'event_' + ComponentType.BUTTON
 * ```
 *
 * @example Concatenation with allowed parameter (returns true):
 * ```typescript
 * function MyComponent({ componentId }: Props) {
 *   return <Component componentId={'prefix_' + componentId} />;
 * }
 * ```
 *
 * @example Invalid: concatenation with function call (returns false):
 * ```typescript
 * 'event_' + generateId() // generateId() is dynamic
 * ```
 *
 * @example Invalid: other operators not supported (returns false):
 * ```typescript
 * eventId - 10     // Subtraction not supported
 * count * 2        // Multiplication not supported
 * value == 'test'  // Comparison not supported
 * ```
 */
function isStaticBinaryExpression(
  node: TSESTree.BinaryExpression,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  if (node.operator === '+') {
    return (
      isStaticValue(node.left, propertyCheck, functionParams, importAliases) &&
      isStaticValue(node.right, propertyCheck, functionParams, importAliases)
    );
  }
  return false;
}

/**
 * Check if a ChainExpression is static.
 *
 * A ChainExpression is a chain of property accesses on an object.
 * It is static if the expression is static.
 *
 * @param node - The ChainExpression AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 * @returns true if the expression is static, false otherwise
 *
 * @example ChainExpression with static property access (returns true):
 * ```typescript
 * const obj = { a: { b: 'c' } };
 * obj.a.b // All parts are static
 * ```
 *
 * @example ChainExpression with dynamic property access (returns false):
 * ```typescript
 * const obj = { a: { b: 'c' } };
 * obj.a.b.d // d is dynamic
 * ```
 */
function isStaticChainExpression(
  node: TSESTree.ChainExpression,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
): boolean {
  return isStaticValue(node.expression, propertyCheck, functionParams, importAliases);
}

/**
 * Check if a name contains any of the allowed enum namespaces, resolving aliases if needed.
 */
function matchesAllowedEnumOrUnionNamespace(
  name: string | null,
  propertyCheck: PropertyCheckConfig,
  importAliases: Map<string, string>,
): boolean {
  if (!name) {
    return false;
  }

  let resolvedName = name;
  const parts = name.split('.');
  if (parts.length > 0) {
    const firstPart = parts[0];
    const aliasedName = importAliases.get(firstPart);
    if (aliasedName) {
      parts[0] = aliasedName;
      resolvedName = parts.join('.');
    }
  }

  const nameParts = resolvedName.split('.');

  return propertyCheck.allowedEnumPrefixesAndTypes.some((namespace) => {
    return nameParts.some((part) => part === namespace || part.startsWith(namespace));
  });
}

/**
 * Type for static value checker functions
 */
type StaticValueChecker = (
  node: any,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
  importAliases: Map<string, string>,
) => boolean;

/**
 * Map of AST node types to their corresponding static value checker functions.
 */
const STATIC_VALUE_CHECKERS: Partial<Record<AST_NODE_TYPES, StaticValueChecker>> = {
  [AST_NODE_TYPES.Literal]: isStaticLiteral,
  [AST_NODE_TYPES.Identifier]: isStaticIdentifier,
  [AST_NODE_TYPES.MemberExpression]: isStaticMemberExpression,
  [AST_NODE_TYPES.TSSatisfiesExpression]: isStaticSatisfiesExpression,
  [AST_NODE_TYPES.ChainExpression]: isStaticChainExpression,
  [AST_NODE_TYPES.TemplateLiteral]: isStaticTemplateLiteral,
  [AST_NODE_TYPES.ConditionalExpression]: isStaticConditionalExpression,
  [AST_NODE_TYPES.LogicalExpression]: isStaticLogicalExpression,
  [AST_NODE_TYPES.BinaryExpression]: isStaticBinaryExpression,
};

/**
 * Check if a node represents a static value that is safe for logging (no PII risk).
 *
 * A value is considered static if it can be determined at code-review time and doesn't
 * contain runtime-dynamic data that could leak PII.
 *
 * @param node - The AST node to check
 * @param propertyCheck - Configuration defining what parameters/enums are allowed
 * @param functionParams - Set of parameter names from enclosing functions
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 *
 * @returns true if the value is static and safe, false otherwise
 *
 * @example Valid cases (returns true):
 * ```typescript
 * // String literals
 * 'static-event-id'
 *
 * // Const variables with static initializers
 * const eventId = 'user_login';
 * recordProto({ observability_log: { entity: { entity_id: eventId } } });
 *
 * // Allowed function parameters (e.g., componentId)
 * function MyComponent({ componentId }: Props) {
 *   recordObservabilityEvent(r, p, { eventEntity: { entityId: componentId } });
 * }
 *
 * // Enum values (direct or via const)
 * ComponentType.BUTTON
 * const source = ComponentType.BUTTON;
 * `foo.${source}.${EnumHere.VALUE}` // source is recognized as enum property
 *
 * // Enum values with import aliases
 * import { ComponentType as CT } from '...';
 * CT.BUTTON // Resolved to ComponentType via importAliases
 *
 * // Template literals with only static parts
 * `user_${eventType}_click` // where eventType is const enum or allowed param
 *
 * // Ternary with static branches
 * isError ? 'error_event' : 'success_event'
 * ```
 *
 * @example Invalid cases (returns false):
 * ```typescript
 * // Function calls (runtime-dynamic)
 * generateId()
 *
 * // let/var variables (mutable, potentially dynamic)
 * let eventId = 'event';
 *
 * // Object property access (could contain user data)
 * props.eventId
 *
 * // Non-allowed parameters
 * function log(userId: string) {
 *   recordProto({ observability_log: { entity: { entity_id: userId } } });
 * }
 *
 * // Template literals with dynamic expressions
 * `user_${computeId()}_event`
 * ```
 */
function isStaticValue(
  node: TSESTree.Node,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string> = new Set(),
  importAliases: Map<string, string> = new Map(),
): boolean {
  const checker = STATIC_VALUE_CHECKERS[node.type];
  if (checker) {
    return checker(node, propertyCheck, functionParams, importAliases);
  }
  return false;
}

/**
 * Check if a function parameter has a type annotation that matches an allowed enum namespace.
 */
function isParameterWithEnumType(
  identifier: TSESTree.Identifier,
  propertyCheck: PropertyCheckConfig,
  importAliases: Map<string, string> = new Map(),
): boolean {
  let current: TSESTree.Node | undefined = identifier.parent;

  while (current) {
    if (
      current.type === AST_NODE_TYPES.FunctionDeclaration ||
      current.type === AST_NODE_TYPES.FunctionExpression ||
      current.type === AST_NODE_TYPES.ArrowFunctionExpression
    ) {
      for (const param of current.params) {
        const paramType = getParameterTypeAnnotation(param, identifier.name);
        if (paramType) {
          const typeName = getTypeNameFromTypeAnnotation(paramType, importAliases);
          if (typeName && matchesAllowedEnumOrUnionNamespace(typeName, propertyCheck, importAliases)) {
            return true;
          }
        }
      }
    }

    current = current.parent;
  }

  return false;
}

/**
 * Get the type annotation for a specific parameter name from a parameter node.
 *
 * Handles:
 * - Simple parameters: `source: EnvironmentConfigurationFormSource`
 * - Object destructuring: `{ source }: {source: EnvironmentConfigurationFormSource}`
 *
 * @param param - The parameter AST node
 * @param paramName - The name of the parameter to find the type for
 * @returns The TSTypeReference if found, null otherwise
 */
function getParameterTypeAnnotation(param: TSESTree.Parameter, paramName: string): TSESTree.TSTypeReference | null {
  if (param.type === AST_NODE_TYPES.Identifier && param.name === paramName) {
    if (param.typeAnnotation && param.typeAnnotation.typeAnnotation.type === AST_NODE_TYPES.TSTypeReference) {
      return param.typeAnnotation.typeAnnotation as TSESTree.TSTypeReference;
    }
  }

  if (param.type === AST_NODE_TYPES.ObjectPattern) {
    for (const prop of param.properties) {
      if (prop.type === AST_NODE_TYPES.Property) {
        let propName: string | null = null;
        if (prop.key.type === AST_NODE_TYPES.Identifier) {
          propName = prop.key.name;
        }
        if (prop.value.type === AST_NODE_TYPES.Identifier) {
          propName = prop.value.name;
        }

        if (propName === paramName) {
          if (param.typeAnnotation && param.typeAnnotation.typeAnnotation.type === AST_NODE_TYPES.TSTypeLiteral) {
            const typeLiteral = param.typeAnnotation.typeAnnotation as TSESTree.TSTypeLiteral;
            for (const typeProp of typeLiteral.members) {
              if (
                typeProp.type === AST_NODE_TYPES.TSPropertySignature &&
                typeProp.key.type === AST_NODE_TYPES.Identifier &&
                typeProp.key.name === paramName
              ) {
                if (
                  typeProp.typeAnnotation &&
                  typeProp.typeAnnotation.typeAnnotation.type === AST_NODE_TYPES.TSTypeReference
                ) {
                  return typeProp.typeAnnotation.typeAnnotation as TSESTree.TSTypeReference;
                }
              }
            }
          }
        }
      }
    }
  }

  if (param.type === AST_NODE_TYPES.AssignmentPattern) {
    return getParameterTypeAnnotation(param.left as TSESTree.Parameter, paramName);
  }

  return null;
}

/**
 * Get the full qualified name from a TSQualifiedName node.
 */
function getQualifiedTypeName(node: TSESTree.TSQualifiedName): string {
  const left =
    node.left.type === AST_NODE_TYPES.Identifier
      ? node.left.name
      : node.left.type === AST_NODE_TYPES.TSQualifiedName
        ? getQualifiedTypeName(node.left)
        : '';
  const right = node.right.type === AST_NODE_TYPES.Identifier ? node.right.name : null;
  return right ? `${left}.${right}` : left;
}

/**
 * Extract the enum name from a TSTypeReference node.
 */
function getTypeNameFromTypeAnnotation(
  typeAnnotation: TSESTree.TSTypeReference,
  importAliases: Map<string, string> = new Map(),
): string | null {
  const typeName = typeAnnotation.typeName;

  if (typeName.type === AST_NODE_TYPES.Identifier) {
    const resolvedName = importAliases.get(typeName.name) || typeName.name;
    return resolvedName;
  } else if (typeName.type === AST_NODE_TYPES.TSQualifiedName) {
    const qualifiedName = getQualifiedTypeName(typeName);
    const parts = qualifiedName.split('.');
    if (parts.length > 0) {
      const firstPart = parts[0];
      const aliasedName = importAliases.get(firstPart);
      if (aliasedName) {
        parts[0] = aliasedName;
        return parts.join('.');
      }
    }
    return qualifiedName;
  }

  return null;
}

/**
 * Check if a MemberExpression is accessing an enum-like value.
 *
 * Enum accesses are considered safe because they represent predefined constant values
 * that don't contain user data.
 *
 * @param node - The MemberExpression AST node to check
 * @param propertyCheck - Configuration with allowed enum namespaces
 * @param importAliases - Map of import aliases (localName -> originalName) for resolving enum names
 *
 * @returns true if the expression accesses an allowed enum value
 *
 * @example Valid enum accesses (returns true):
 * ```typescript
 * ComponentType.BUTTON
 * EntityType.USER
 * EventType.CLICK
 * ProductEntityType.NOTEBOOK
 *
 * // With import alias:
 * import { ComponentType as CT } from '...';
 * CT.BUTTON // Resolved to ComponentType
 * ```
 *
 * @example Invalid (returns false):
 * ```typescript
 * props.eventId      // Not an enum
 * config.defaultId   // Not an enum
 * ```
 */
function isEnumAccess(
  node: TSESTree.MemberExpression,
  propertyCheck: PropertyCheckConfig,
  importAliases: Map<string, string> = new Map(),
): boolean {
  const objectName = getObjectName(node.object);
  return matchesAllowedEnumOrUnionNamespace(objectName, propertyCheck, importAliases);
}

/**
 * Get the full name of an object from nested MemberExpressions.
 *
 * Recursively traverses MemberExpression chains to build the full dotted path.
 * Used primarily for identifying enum accesses.
 *
 * @param node - The AST node to extract the name from
 *
 * @returns The full dotted name, or null if not extractable
 *
 * @example
 * ```typescript
 * ComponentType        // Returns: 'ComponentType'
 * Foo.Bar.Baz         // Returns: 'Foo.Bar.Baz'
 * ```
 */
function getObjectName(node: TSESTree.Node): string | null {
  if (node.type === AST_NODE_TYPES.Identifier) {
    return node.name;
  }
  if (node.type === AST_NODE_TYPES.MemberExpression) {
    const left = getObjectName(node.object);
    const right = node.property.type === AST_NODE_TYPES.Identifier ? node.property.name : null;
    if (left && right) {
      return `${left}.${right}`;
    }
  }
  return null;
}

/**
 * Navigate through nested object properties to find a specific property value.
 *
 * Given an ObjectExpression and a path (array of property names), this function
 * traverses the object tree to find the value at that path.
 *
 * @param obj - The ObjectExpression AST node to search in
 * @param propertyPath - Array of property names forming the path to traverse
 *
 * @returns The AST node representing the value, or null if not found
 *
 * @example
 * ```typescript
 * // Given: { observability_log: { entity: { entity_id: 'user_login' } } }
 * getPropertyValue(obj, ['observability_log', 'entity', 'entity_id'])
 * // Returns the node for: 'user_login'
 *
 * // Given: { observability_log: { entity: entityVar } }
 * getPropertyValue(obj, ['observability_log', 'entity'])
 * // Returns the node for: entityVar
 * ```
 */
function getPropertyValue(obj: TSESTree.ObjectExpression, propertyPath: string[]): TSESTree.Node | null {
  if (propertyPath.length === 0) {
    return null;
  }

  const [currentKey, ...remainingPath] = propertyPath;

  for (const prop of obj.properties) {
    if (prop.type === AST_NODE_TYPES.Property && !prop.computed) {
      const key =
        prop.key.type === AST_NODE_TYPES.Identifier
          ? prop.key.name
          : prop.key.type === AST_NODE_TYPES.Literal
            ? String(prop.key.value)
            : null;

      if (key === currentKey) {
        if (remainingPath.length === 0) {
          return prop.value;
        }
        if (prop.value.type === AST_NODE_TYPES.ObjectExpression) {
          return getPropertyValue(prop.value, remainingPath);
        }
      }
    }
  }

  return null;
}

/**
 * Find the matching function configuration for a call expression.
 */
function getMatchingConfig(node: TSESTree.CallExpression): FunctionConfig | null {
  const callee = node.callee;

  if (callee.type === AST_NODE_TYPES.Identifier) {
    const name = callee.name;
    return FUNCTION_CONFIGS.find((config) => !config.className && config.functionNames.includes(name)) || null;
  }

  if (
    callee.type === AST_NODE_TYPES.MemberExpression &&
    callee.object.type === AST_NODE_TYPES.Identifier &&
    callee.property.type === AST_NODE_TYPES.Identifier
  ) {
    const className = callee.object.name;
    const methodName = callee.property.name;
    return (
      FUNCTION_CONFIGS.find((config) => config.className === className && config.functionNames.includes(methodName)) ||
      null
    );
  }

  return null;
}

/**
 * Check if a parameter comes from a React hook function (starts with "use").
 *
 * We disallow parameters from hooks as we only validate React component parameters.
 *
 * @param paramName - The parameter name to check
 * @param node - The AST node to start searching from
 * @returns true if the parameter is defined in a hook function
 *
 * @example Hook function (returns true):
 * ```typescript
 * function useMyHook(componentId: string) {
 *   return <Component componentId={componentId} />; // NOT ALLOWED
 * }
 * ```
 *
 * @example Regular React component (returns false):
 * ```typescript
 * // This is a component, not a hook
 * function MyComponent({ componentId }: Props) {
 *   // componentId here is from a component - ALLOWED
 *   return <Component componentId={componentId} />;
 * }
 * ```
 */
function isParameterFromHook(paramName: string, node: TSESTree.Node): boolean {
  let current: TSESTree.Node | undefined = node;

  while (current) {
    if (
      (current.type === AST_NODE_TYPES.FunctionDeclaration || current.type === AST_NODE_TYPES.FunctionExpression) &&
      current.id?.name.startsWith('use')
    ) {
      for (const param of current.params) {
        const paramNames = extractParamNames(param);
        if (paramNames.includes(paramName)) {
          return true;
        }
      }
    }

    if (
      current.type === AST_NODE_TYPES.ArrowFunctionExpression &&
      current.parent?.type === AST_NODE_TYPES.VariableDeclarator &&
      current.parent.id.type === AST_NODE_TYPES.Identifier &&
      current.parent.id.name.startsWith('use')
    ) {
      for (const param of current.params) {
        const paramNames = extractParamNames(param);
        if (paramNames.includes(paramName)) {
          return true;
        }
      }
    }

    current = current.parent;
  }

  return false;
}

/**
 * Check if an identifier is a const variable with a static initializer.
 *
 * Only allows 'const' (not 'let' or 'var') since mutable variables can be reassigned.
 *
 * @param node - The Identifier AST node to check
 * @param propertyCheck - Configuration for validating the static value
 * @param importAliases - Map of import aliases for resolving enum names
 * @returns true if the identifier is a const with a static initializer
 *
 * @example Valid (returns true):
 * ```typescript
 * const eventId = 'user_login';
 * recordProto({ observability_log: { entity: { entity_id: eventId } } });
 * ```
 *
 * @example Invalid: let variable (returns false):
 * ```typescript
 * let eventId = 'user_login'; // Mutable
 * ```
 *
 * @example Invalid: const with dynamic initializer (returns false):
 * ```typescript
 * const eventId = generateId(); // Dynamic
 * ```
 */
function isConstWithStaticValue(
  node: TSESTree.Identifier,
  propertyCheck: PropertyCheckConfig,
  importAliases: Map<string, string> = new Map(),
): boolean {
  // Walk up the AST to find variable declarations
  let current: TSESTree.Node | undefined = node.parent;

  while (current) {
    // Check VariableDeclaration nodes at any scope level (function, module, block, etc.)
    if (current.type === AST_NODE_TYPES.VariableDeclaration) {
      // Only allow const declarations
      if (current.kind === 'const') {
        for (const declarator of current.declarations) {
          if (declarator.id.type === AST_NODE_TYPES.Identifier && declarator.id.name === node.name && declarator.init) {
            // Recursively check if the initializer is static
            return isStaticValue(declarator.init, propertyCheck, new Set(), importAliases);
          }
        }
      }
    }

    // For BlockStatements (function bodies, if statements, etc.) and Program nodes (module-level),
    // search through sibling statements because const declarations and their usages are siblings in the AST
    if (current.type === AST_NODE_TYPES.BlockStatement || current.type === AST_NODE_TYPES.Program) {
      for (const statement of current.body) {
        // Check direct VariableDeclaration statements
        // e.g., const MY_CONST = 'value';
        if (statement.type === AST_NODE_TYPES.VariableDeclaration && statement.kind === 'const') {
          for (const declarator of statement.declarations) {
            if (
              declarator.id.type === AST_NODE_TYPES.Identifier &&
              declarator.id.name === node.name &&
              declarator.init
            ) {
              // Pass importAliases so enum detection works correctly
              return isStaticValue(declarator.init, propertyCheck, new Set(), importAliases);
            }
          }
        }

        // Check ExportNamedDeclaration statements that contain VariableDeclaration
        // e.g., export const MY_CONST = 'value';
        if (
          statement.type === AST_NODE_TYPES.ExportNamedDeclaration &&
          statement.declaration?.type === AST_NODE_TYPES.VariableDeclaration &&
          statement.declaration.kind === 'const'
        ) {
          for (const declarator of statement.declaration.declarations) {
            if (
              declarator.id.type === AST_NODE_TYPES.Identifier &&
              declarator.id.name === node.name &&
              declarator.init
            ) {
              // Pass importAliases so enum detection works correctly
              return isStaticValue(declarator.init, propertyCheck, new Set(), importAliases);
            }
          }
        }
      }
    }

    current = current.parent;
  }

  return false;
}

/**
 * Check if a VariableDeclarator destructures a property from a function parameter.
 *
 * Matches pattern: `const { targetName } = paramName` where paramName is a function parameter.
 *
 * @param declarator - The VariableDeclarator AST node to check
 * @param targetName - The property name to look for
 * @param functionParams - Set of function parameter names
 * @param propertyCheck - Configuration with allowed parameters
 * @returns true if this declarator destructures targetName from a parameter
 *
 * @example Valid (simple param):
 * ```typescript
 * function MyComponent(componentId: string) {
 *   const { componentId } = componentId; // Unusual but matches simple param
 * }
 * ```
 *
 * @example Valid (object-property param):
 * ```typescript
 * function MyComponent(props: Props) {
 *   const { componentId } = props; // props is at arg 0, destructuring componentId
 * }
 * ```
 *
 * @example Invalid (not from parameter):
 * ```typescript
 * const obj = { componentId: 'id' };
 * const { componentId } = obj; // obj is not a parameter
 * ```
 */
function isDestructuringAllowedParam(
  declarator: TSESTree.VariableDeclarator,
  targetName: string,
  functionParams: Set<string>,
  propertyCheck: PropertyCheckConfig,
): boolean {
  if (declarator.id.type !== AST_NODE_TYPES.ObjectPattern || declarator.init?.type !== AST_NODE_TYPES.Identifier) {
    return false;
  }

  const sourceParam = declarator.init.name;
  if (!functionParams.has(sourceParam)) {
    return false;
  }

  if (isParameterFromHook(sourceParam, declarator)) {
    return false;
  }

  for (const prop of declarator.id.properties) {
    if (prop.type === AST_NODE_TYPES.Property) {
      const propValue =
        prop.value.type === AST_NODE_TYPES.Identifier
          ? prop.value.name
          : prop.value.type === AST_NODE_TYPES.AssignmentPattern && prop.value.left.type === AST_NODE_TYPES.Identifier
            ? prop.value.left.name
            : null;

      if (propValue === targetName) {
        const propKey =
          prop.key.type === AST_NODE_TYPES.Identifier
            ? prop.key.name
            : prop.key.type === AST_NODE_TYPES.Literal
              ? String(prop.key.value)
              : null;

        if (propKey === targetName) {
          for (const paramConfig of propertyCheck.allowedParams) {
            if (paramConfig.type === 'object-property') {
              if (
                isParameterAtPosition(declarator, sourceParam, paramConfig.argPosition) &&
                arraysEqual(paramConfig.propertyPath, [targetName])
              ) {
                return true;
              }
            }
          }
        }
      }
    }
  }

  return false;
}

/**
 * Check if a parameter name is at a specific argument position in an enclosing function.
 */
function isParameterAtPosition(node: TSESTree.Node, paramName: string, position: number): boolean {
  let current: TSESTree.Node | undefined = node;

  while (current) {
    if (
      current.type === AST_NODE_TYPES.FunctionDeclaration ||
      current.type === AST_NODE_TYPES.FunctionExpression ||
      current.type === AST_NODE_TYPES.ArrowFunctionExpression
    ) {
      const param = current.params[position];
      if (!param) {
        current = current.parent;
        continue;
      }

      if (param.type === AST_NODE_TYPES.Identifier && param.name === paramName) {
        return true;
      }

      if (
        param.type === AST_NODE_TYPES.AssignmentPattern &&
        param.left.type === AST_NODE_TYPES.Identifier &&
        param.left.name === paramName
      ) {
        return true;
      }
    }

    current = current.parent;
  }

  return false;
}

/**
 * Check if an identifier is destructured from an allowed function parameter.
 *
 * Handles React component patterns like `const { componentId } = props`.
 *
 * @param node - The Identifier AST node being checked
 * @param propertyCheck - Configuration with allowed parameter names
 * @param functionParams - Set of parameter names from enclosing functions
 * @returns true if the identifier is destructured from an allowed parameter
 *
 * @example Valid (returns true):
 * ```typescript
 * function MyComponent(props: { componentId: string }) {
 *   const { componentId } = props;
 *   return <Component componentId={componentId} />;
 * }
 * ```
 *
 * @example Invalid: not from parameter (returns false):
 * ```typescript
 * const props = { componentId: 'id' };
 * const { componentId } = props;
 * ```
 */
function isDestructuredFromAllowedParam(
  node: TSESTree.Identifier,
  propertyCheck: PropertyCheckConfig,
  functionParams: Set<string>,
): boolean {
  const matchesObjectPropertyPath = propertyCheck.allowedParams.some(
    (config) =>
      config.type === 'object-property' && config.propertyPath.length === 1 && config.propertyPath[0] === node.name,
  );

  if (!matchesObjectPropertyPath) {
    return false;
  }

  let current: TSESTree.Node | undefined = node.parent;

  while (current) {
    if (current.type === AST_NODE_TYPES.VariableDeclarator) {
      if (isDestructuringAllowedParam(current, node.name, functionParams, propertyCheck)) {
        return true;
      }
    }

    if (current.type === AST_NODE_TYPES.BlockStatement || current.type === AST_NODE_TYPES.Program) {
      for (const statement of current.body) {
        if (statement.type === AST_NODE_TYPES.VariableDeclaration) {
          for (const declarator of statement.declarations) {
            if (isDestructuringAllowedParam(declarator, node.name, functionParams, propertyCheck)) {
              return true;
            }
          }
        }
      }
    }

    current = current.parent;
  }

  return false;
}

/**
 * Extract parameter names from a parameter node, handling destructuring patterns.
 *
 * TypeScript/JavaScript supports multiple parameter patterns (simple, destructured, with defaults).
 * This function recursively extracts all variable names that will be bound in the function scope.
 *
 * @param param - The parameter AST node to extract names from
 * @returns Array of parameter names available in the function scope
 *
 * @example Simple:
 * ```typescript
 * function foo(componentId: string) // ['componentId']
 * ```
 *
 * @example Object destructuring:
 * ```typescript
 * function foo({ componentId, eventId }: Props) // ['componentId', 'eventId']
 * function foo({ id: componentId }: Props) // ['componentId']
 * ```
 *
 * @example Array destructuring:
 * ```typescript
 * function foo([first, second]: string[]) // ['first', 'second']
 * ```
 */
function extractParamNames(param: TSESTree.Parameter): string[] {
  const names: string[] = [];

  if (param.type === AST_NODE_TYPES.Identifier) {
    names.push(param.name);
  } else if (param.type === AST_NODE_TYPES.AssignmentPattern) {
    return extractParamNames(param.left as TSESTree.Parameter);
  } else if (param.type === AST_NODE_TYPES.ObjectPattern) {
    for (const prop of param.properties) {
      if (prop.type === AST_NODE_TYPES.Property) {
        if (prop.value.type === AST_NODE_TYPES.Identifier) {
          names.push(prop.value.name);
        } else if (
          prop.value.type === AST_NODE_TYPES.AssignmentPattern &&
          prop.value.left.type === AST_NODE_TYPES.Identifier
        ) {
          names.push(prop.value.left.name);
        }
      } else if (prop.type === AST_NODE_TYPES.RestElement && prop.argument.type === AST_NODE_TYPES.Identifier) {
        names.push(prop.argument.name);
      }
    }
  } else if (param.type === AST_NODE_TYPES.ArrayPattern) {
    for (const element of param.elements) {
      if (element && element.type === AST_NODE_TYPES.Identifier) {
        names.push(element.name);
      } else if (
        element &&
        element.type === AST_NODE_TYPES.AssignmentPattern &&
        element.left.type === AST_NODE_TYPES.Identifier
      ) {
        names.push(element.left.name);
      } else if (
        element &&
        element.type === AST_NODE_TYPES.RestElement &&
        element.argument.type === AST_NODE_TYPES.Identifier
      ) {
        names.push(element.argument.name);
      }
    }
  } else if (param.type === AST_NODE_TYPES.RestElement && param.argument.type === AST_NODE_TYPES.Identifier) {
    names.push(param.argument.name);
  }

  return names;
}

/**
 * Get parameter names from all enclosing functions.
 */
function getFunctionParams(node: TSESTree.Node): Set<string> {
  const params = new Set<string>();

  let current: TSESTree.Node | undefined = node;
  while (current) {
    if (
      current.type === AST_NODE_TYPES.FunctionDeclaration ||
      current.type === AST_NODE_TYPES.FunctionExpression ||
      current.type === AST_NODE_TYPES.ArrowFunctionExpression
    ) {
      for (const param of current.params) {
        const paramNames = extractParamNames(param);
        for (const name of paramNames) {
          params.add(name);
        }
      }
    }
    current = current.parent;
  }

  return params;
}

/**
 * Get a human-readable display name for a function call (for error messages).
 */
function getFunctionDisplayName(node: TSESTree.CallExpression, config: FunctionConfig): string {
  const callee = node.callee;

  if (callee.type === AST_NODE_TYPES.Identifier) {
    return callee.name;
  }

  if (
    callee.type === AST_NODE_TYPES.MemberExpression &&
    callee.object.type === AST_NODE_TYPES.Identifier &&
    callee.property.type === AST_NODE_TYPES.Identifier
  ) {
    return `${callee.object.name}.${callee.property.name}`;
  }

  return config.className ? `${config.className}.<method>` : '<function>';
}

/**
 * Validate a call expression to ensure property values are static.
 */
function validateCallExpression(
  node: TSESTree.CallExpression,
  context: ReturnType<typeof createRuleWithoutOptions<MessageIds>>['create'] extends (context: infer C) => any
    ? C
    : never,
  importAliases: Map<string, string>,
  hookReturnedFunctions: Map<string, string>,
): void {
  let resolvedName: string | null = null;

  if (node.callee.type === AST_NODE_TYPES.Identifier) {
    const calleeName = node.callee.name;

    const originalImportName = importAliases.get(calleeName);
    if (originalImportName) {
      resolvedName = originalImportName;
    }

    const hookFunctionName = hookReturnedFunctions.get(calleeName);
    if (hookFunctionName) {
      resolvedName = hookFunctionName;
    }

    if (!resolvedName) {
      resolvedName = calleeName;
    }
  }

  const config = resolvedName
    ? // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      FUNCTION_CONFIGS.find((cfg) => !cfg.className && cfg.functionNames.includes(resolvedName!)) ||
      getMatchingConfig(node)
    : getMatchingConfig(node);

  if (!config) {
    return;
  }

  const functionParams = getFunctionParams(node);
  const argIndex = config.getArgumentIndex(node);
  const argToCheck = node.arguments[argIndex];
  const functionName = getFunctionDisplayName(node, config);

  if (!argToCheck) {
    return;
  }

  for (const propertyCheck of config.propertyChecks) {
    if (config.requireInlineObject && argToCheck.type !== AST_NODE_TYPES.ObjectExpression) {
      const propertyPathInfo =
        propertyCheck.propertyPath.length > 0 ? ` (checking property: ${propertyCheck.propertyPath.join('.')})` : '';

      context.report({
        node: argToCheck,
        messageId: 'propertyValueFromVariable',
        data: {
          functionName,
          objectType: 'Argument object',
          propertyPathInfo,
        },
      });
      return;
    }

    if (propertyCheck.propertyPath.length === 0) {
      if (!isStaticValue(argToCheck, propertyCheck, functionParams, importAliases)) {
        const enumInfo =
          propertyCheck.allowedEnumPrefixesAndTypes.length > 0
            ? `, enum namespaces (${propertyCheck.allowedEnumPrefixesAndTypes.join(', ')})`
            : '';
        const paramInfo =
          propertyCheck.allowedParams.length > 0
            ? `, allowed parameters (${propertyCheck.allowedParams
                .map((p) => (p.type === 'simple' ? p.name : `${p.argPosition}:${p.propertyPath.join('.')}`))
                .join(', ')})`
            : '';

        context.report({
          node: argToCheck,
          messageId: 'dynamicPropertyValue',
          data: {
            functionName,
            propertyPath: `argument ${argIndex + 1}`,
            enumInfo,
            paramInfo,
          },
        });
      }
      continue;
    }

    if (argToCheck.type !== AST_NODE_TYPES.ObjectExpression) {
      continue;
    }

    const parentPath = propertyCheck.propertyPath.slice(0, -1);
    if (parentPath.length > 0) {
      const parentValue = getPropertyValue(argToCheck, parentPath);
      if (parentValue && parentValue.type === AST_NODE_TYPES.Identifier) {
        context.report({
          node: parentValue,
          messageId: 'propertyValueFromVariable',
          data: {
            functionName,
            objectType: `Object "${parentPath.join('.')}"`,
            propertyPathInfo: ` (parent of: ${propertyCheck.propertyPath.join('.')})`,
          },
        });
        return;
      }
    }

    const propertyValue = getPropertyValue(argToCheck, propertyCheck.propertyPath);
    if (propertyValue && !isStaticValue(propertyValue, propertyCheck, functionParams, importAliases)) {
      const enumInfo =
        propertyCheck.allowedEnumPrefixesAndTypes.length > 0
          ? `, enum namespaces (${propertyCheck.allowedEnumPrefixesAndTypes.join(', ')})`
          : '';
      const paramInfo =
        propertyCheck.allowedParams.length > 0
          ? `, allowed parameters (${propertyCheck.allowedParams
              .map((p) => (p.type === 'simple' ? p.name : `${p.argPosition}:${p.propertyPath.join('.')}`))
              .join(', ')})`
          : '';

      context.report({
        node: propertyValue,
        messageId: 'dynamicPropertyValue',
        data: {
          functionName,
          propertyPath: propertyCheck.propertyPath.join('.'),
          enumInfo,
          paramInfo,
        },
      });
    }
  }
}

/**
 * Validate a JSX attribute to ensure its value is static.
 */
function validateJSXAttribute(
  node: TSESTree.JSXAttribute,
  context: ReturnType<typeof createRuleWithoutOptions<MessageIds>>['create'] extends (context: infer C) => any
    ? C
    : never,
  importAliases: Map<string, string> = new Map(),
): void {
  if (node.name.type !== AST_NODE_TYPES.JSXIdentifier || !node.value) {
    return;
  }

  const config = JSX_ATTRIBUTE_CONFIGS.find((cfg) => cfg.attributeName === node.name.name);
  if (!config) {
    return;
  }

  if (config.componentNameList && config.componentNameList.length > 0) {
    let current: TSESTree.Node | undefined = node.parent;
    while (current) {
      if (current.type === AST_NODE_TYPES.JSXOpeningElement) {
        const elementName = current.name;
        let componentName: string | undefined;

        if (elementName.type === AST_NODE_TYPES.JSXIdentifier) {
          componentName = elementName.name;
        } else if (elementName.type === AST_NODE_TYPES.JSXMemberExpression) {
          let memberExpr = elementName;
          while (memberExpr.property.type === AST_NODE_TYPES.JSXIdentifier) {
            componentName = memberExpr.property.name;
            if (memberExpr.object.type === AST_NODE_TYPES.JSXIdentifier) {
              componentName = `${memberExpr.object.name}.${componentName}`;
              break;
            } else if (memberExpr.object.type === AST_NODE_TYPES.JSXMemberExpression) {
              memberExpr = memberExpr.object;
            } else {
              break;
            }
          }
        }

        if (!componentName || !config.componentNameList.includes(componentName)) {
          return;
        }
        break;
      }
      current = current.parent as TSESTree.Node | undefined;
    }
  }

  const functionParams = getFunctionParams(node);
  const propertyCheck = config.propertyCheck;

  if (node.value.type === AST_NODE_TYPES.Literal) {
    return;
  }

  if (node.value.type === AST_NODE_TYPES.JSXExpressionContainer) {
    const expr = node.value.expression;
    if (
      expr.type !== AST_NODE_TYPES.JSXEmptyExpression &&
      !isStaticValue(expr, propertyCheck, functionParams, importAliases)
    ) {
      const enumInfo =
        propertyCheck.allowedEnumPrefixesAndTypes.length > 0
          ? `, enum namespaces (${propertyCheck.allowedEnumPrefixesAndTypes.join(', ')})`
          : '';
      const paramInfo =
        propertyCheck.allowedParams.length > 0
          ? `, allowed parameters (${propertyCheck.allowedParams
              .map((p) => (p.type === 'simple' ? p.name : `${p.argPosition}:${p.propertyPath.join('.')}`))
              .join(', ')})`
          : '';

      context.report({
        node: node.value,
        messageId: 'dynamicPropertyValue',
        data: {
          functionName: 'JSX Component',
          propertyPath: config.attributeName,
          enumInfo,
          paramInfo,
        },
      });
    }
  }
}

export default createRuleWithoutOptions<MessageIds>({
  name: 'no-dynamic-property-value',
  meta: {
    type: 'problem',
    docs: {
      description: 'Enforce static property values to ensure no PII is logged.',
    },
    messages: {
      dynamicPropertyValue:
        '{{ functionName }}: Property "{{ propertyPath }}" must be static. Allowed: string literals{{ enumInfo }}{{ paramInfo }}, const variables with static values, or combinations via template strings. Avoid: let/var variables, function calls, or complex expressions.',
      propertyValueFromVariable:
        '{{ functionName }}: {{ objectType }} must be defined inline, not from a variable. This ensures property values can be statically verified. Define the object directly in the function call{{ propertyPathInfo }}.',
    },
    fixable: undefined,
  },
  create(context) {
    // Track import aliases: aliasName -> originalName
    const importAliases = new Map<string, string>();

    // Track variables that are assigned from hook calls: variableName -> hookName
    const hookReturnedFunctions = new Map<string, string>();

    return {
      // Track import declarations to handle aliases
      ImportDeclaration(node: TSESTree.ImportDeclaration) {
        for (const specifier of node.specifiers) {
          if (specifier.type === AST_NODE_TYPES.ImportSpecifier) {
            const importedName =
              specifier.imported.type === AST_NODE_TYPES.Identifier
                ? specifier.imported.name
                : specifier.imported.value;
            const localName = specifier.local.name;
            if (importedName !== localName) {
              // Store the alias mapping: local name -> imported name
              importAliases.set(localName, String(importedName));
            }
          }
        }
      },

      // Track variable declarations from hook calls
      VariableDeclarator(node: TSESTree.VariableDeclarator) {
        if (
          node.id.type === AST_NODE_TYPES.Identifier &&
          node.init?.type === AST_NODE_TYPES.CallExpression &&
          node.init.callee.type === AST_NODE_TYPES.Identifier
        ) {
          const calleeName = node.init.callee.name;
          // Resolve alias if it exists
          const originalName = importAliases.get(calleeName) || calleeName;

          // Check if this is a hook call (starts with 'use')
          if (originalName.startsWith('use')) {
            // Map the variable name to the hook's corresponding function name
            // e.g., useRecordObservabilityEvent -> recordObservabilityEvent
            const functionName = originalName.replace(/^use/, '').replace(/^./, (c) => c.toLowerCase());
            hookReturnedFunctions.set(node.id.name, functionName);
          }
        }
      },

      CallExpression(node: TSESTree.CallExpression) {
        validateCallExpression(node, context, importAliases, hookReturnedFunctions);
      },

      JSXAttribute(node: TSESTree.JSXAttribute) {
        validateJSXAttribute(node, context, importAliases);
      },
    };
  },
});
