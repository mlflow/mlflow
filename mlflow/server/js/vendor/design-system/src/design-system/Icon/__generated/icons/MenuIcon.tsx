import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgMenuIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M15 4H1V2.5h14zm0 4.75H1v-1.5h14zm0 4.75H1V12h14z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const MenuIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgMenuIcon} />;
});
MenuIcon.displayName = 'MenuIcon';
export default MenuIcon;
