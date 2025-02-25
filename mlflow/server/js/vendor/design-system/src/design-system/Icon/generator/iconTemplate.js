function removeClipPathIds(element) {
  if (element.type === 'JSXElement' && element.openingElement.name.name === 'clipPath') {
    element.openingElement.attributes = element.openingElement.attributes.filter((attr) => attr.name.name !== 'id');
  }

  if (element.children) {
    element.children.forEach((child) => removeClipPathIds(child));
  }
}

const colorExclusionList = ['none', '#fff'];
const exclusionRegex = new RegExp(
  `^(?!(${colorExclusionList.map((color) => color.replace(/[#]/g, '\\#')).join('|')})$).*`,
);

function replaceFillandStroke(element) {
  if (element.type === 'JSXElement') {
    element.openingElement.attributes = element.openingElement.attributes.map((attr) => {
      if (
        (attr.name.name === 'fill' || attr.name.name === 'stroke') &&
        attr.value &&
        exclusionRegex.test(attr.value.value)
      ) {
        return { ...attr, value: { type: 'StringLiteral', value: 'currentColor' } };
      }
      return attr;
    });
  }
  if (element.children) {
    element.children.forEach((child) => replaceFillandStroke(child));
  }
}

function defaultTemplate({ imports, interfaces, componentName, props, jsx }, { tpl, options: opts }) {
  const antIconName = componentName.substring(3);
  const filteredImports = imports.filter((item) => item.specifiers[0].local.name !== 'SVGProps');

  removeClipPathIds(jsx);
  replaceFillandStroke(jsx);
  return tpl`
${filteredImports}
  import { forwardRef } from 'react';
  import type { Ref } from 'react';

  
  import type { IconProps } from '../../Icon';
  import { Icon } from '../../Icon';


${interfaces}

function ${componentName}(props: React.SVGProps<SVGSVGElement>) {
  return ${jsx};
}

const ${antIconName} = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
    return <Icon ref={forwardedRef} {...props} component={${componentName}} />;
});

${antIconName}.displayName = "${antIconName}";

export default ${antIconName};`;
}
module.exports = defaultTemplate;
