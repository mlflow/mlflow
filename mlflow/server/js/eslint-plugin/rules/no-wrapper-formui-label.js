const forbiddenChildrenTypes = new Set([
  'input',
  'select',
  'textarea',
  'checkbox',
  'radio',
  'autocomplete',
  'dialogcombobox',
  'segmentedcontrolgroup',
  'selectv2',
  'switch',
  'togglebutton',
  'typeaheadcomboboxroot',
]);

const findForbiddenDescendant = (node) => {
  if (node.children) {
    for (const child of node.children) {
      const childNodeName = child.openingElement?.name?.object?.name ?? child.openingElement?.name?.name ?? '';
      if (child.type === 'JSXElement' && childNodeName) {
        if (forbiddenChildrenTypes.has(childNodeName.toLowerCase())) {
          return childNodeName;
        }

        return findForbiddenDescendant(child);
      }
    }
  }
};

module.exports = {
  meta: {
    type: 'problem',
    messages: {
      htmlForAttributeMissing:
        'DuBois: Missing "htmlFor" attribute on FormUI.Label. Use "htmlFor" attribute instead of wrapping input elements with FormUI.Label',
      formUILabelWrappingInput:
        'DuBois: FormUI.Label should not wrap {{ name }} elements. Use "htmlFor" attribute instead of wrapping input elements with FormUI.Label',
    },
  },
  create(context) {
    return {
      // Check for missing htmlFor attribute on FormUI.Label
      JSXOpeningElement(node) {
        if (
          node.name.type === 'JSXMemberExpression' &&
          node.name.object.name === 'FormUI' &&
          node.name.property.name === 'Label'
        ) {
          const htmlForAttribute = node.attributes.find((attribute) => attribute.name.name === 'htmlFor');

          if (!htmlForAttribute) {
            context.report({
              node,
              messageId: 'htmlForAttributeMissing',
            });
          }

          // Check if FormUI.Label is wrapping a forbidden form element type
          const forbiddenDescendant = findForbiddenDescendant(node.parent);
          if (forbiddenDescendant) {
            context.report({
              node,
              messageId: 'formUILabelWrappingInput',
              data: {
                name: forbiddenDescendant,
              },
            });
          }
        }
      },
    };
  },
};
