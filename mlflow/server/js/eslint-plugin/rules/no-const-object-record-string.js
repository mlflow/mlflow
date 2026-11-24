const ERROR = 'NO_CONST_OBJECT_RECORD_STRING';

module.exports = {
  meta: {
    fixable: 'code',
    messages: {
      [ERROR]:
        'Do not use `Record<string, *> when declaring a non-empty constant object because it will remove type checking of properties, resulting in unsafe access. Either specify a stricter type for the keys of the object or use `satisfies Record<string, *>` if you want to infer the keys.',
    },
  },

  create(context) {
    const sourceCode = context.getSourceCode();

    return {
      VariableDeclarator(node) {
        if (
          node.init?.type !== 'ObjectExpression' ||
          // Ignore empty objects because their type cannot be inferred
          node.init?.properties?.length === 0 ||
          node.id?.typeAnnotation?.typeAnnotation === undefined ||
          node.id?.typeAnnotation?.typeAnnotation?.typeName?.name !== 'Record' ||
          node.id?.typeAnnotation?.typeAnnotation?.typeArguments?.params?.[0]?.type !== 'TSStringKeyword'
        ) {
          return;
        }

        const typeAnnotation = node.id.typeAnnotation;
        const typeAnnotationText = sourceCode.getText(typeAnnotation.typeAnnotation);

        context.report({
          fix(fixer) {
            return [
              // Remove the `: Record<string, *>` type annotation
              fixer.remove(typeAnnotation),
              // Add the ` satisfies Record<string, *>` type annotation
              fixer.insertTextAfter(node, ` satisfies ${typeAnnotationText}`),
            ];
          },
          node,
          messageId: ERROR,
        });
      },
    };
  },
};
