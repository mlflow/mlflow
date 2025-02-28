import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgItalicIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M9.648 4.5H12V3H6v1.5h2.102l-1.75 7H4V13h6v-1.5H7.898z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ItalicIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgItalicIcon} />;
});
ItalicIcon.displayName = 'ItalicIcon';
export default ItalicIcon;
