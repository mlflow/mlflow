import type { API, FileInfo } from 'jscodeshift';
import _ from 'lodash';

const isKebabCase = (s: string): boolean => s === _.kebabCase(s);

/**
 * Codemod to transform all kebab-case object keys in a file to camelCase. This works for the following cases:
 *
 * 1. Object properties, eg. converting `{ 'bar-baz': 'foo' }` to `{ barBaz: 'foo' }`
 * 2. TS interface properties, eg. converting: `{ 'bar-baz': string; }` to `{ barBaz: string; }`
 * 3. Object member expressions, eg. converting `foo['bar-baz']` to `foo.barBaz`
 *
 * @param file JSCodeShift file object
 * @param api JSCodeShift API object
 * @returns Source code of the transformed file.
 */
export default function transform(file: FileInfo, api: API): string {
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('transforming', file.path);

  const j = api.jscodeshift;
  const root = j(file.source);

  // Convert case 1: object properties
  root
    .find(j.ObjectProperty)
    .filter(({ node }) => node.key.type === 'StringLiteral' && isKebabCase(node.key.value))
    .replaceWith((nodePath) => {
      const { node } = nodePath;
      if (node.key.type === 'StringLiteral') {
        node.key = j.identifier(_.camelCase(node.key.value));
        node.computed = false;
      }
      return node;
    });

  // Convert case 2: TS interface properties
  root
    .find(j.TSPropertySignature)
    .filter(({ node }) => {
      return node.key.type === 'StringLiteral' && isKebabCase(node.key.value);
    })
    .replaceWith((nodePath) => {
      const { node } = nodePath;
      if (node.key.type === 'StringLiteral') {
        node.key = j.identifier(_.camelCase(node.key.value));
      }
      return node;
    });

  // Convert case 3: object member expressions
  root
    .find(j.MemberExpression, {
      property: {
        type: 'StringLiteral',
      },
    })
    .filter((nodePath) => nodePath.node.property.type === 'StringLiteral' && isKebabCase(nodePath.node.property.value))
    .replaceWith((exp) => {
      const { node } = exp;

      if (node.property.type !== 'StringLiteral') {
        return node;
      }

      node.property = j.identifier(_.camelCase(node.property.value));
      node.computed = false;
      return node;
    });

  return root.toSource();
}
