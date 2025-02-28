import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgSidebarAutoIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 17 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z"
        clipRule="evenodd"
      />
      <path
        fill="currentColor"
        d="m9.06 8 1.97 1.97-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06zM11.97 6.03 13.94 8l-1.97 1.97 1.06 1.06L16.06 8l-3.03-3.03z"
      />
    </svg>
  );
}
const SidebarAutoIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgSidebarAutoIcon} />;
});
SidebarAutoIcon.displayName = 'SidebarAutoIcon';
export default SidebarAutoIcon;
