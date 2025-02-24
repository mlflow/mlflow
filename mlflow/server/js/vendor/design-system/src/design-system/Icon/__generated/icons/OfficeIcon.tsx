import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgOfficeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M4 8.75h8v-1.5H4zM7 5.75H4v-1.5h3zM4 11.75h8v-1.5H4z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V5a.75.75 0 0 0-.75-.75H10v-2.5A.75.75 0 0 0 9.25 1zm.75 1.5h6V5c0 .414.336.75.75.75h4.25v7.75h-11z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const OfficeIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgOfficeIcon} />;
});
OfficeIcon.displayName = 'OfficeIcon';
export default OfficeIcon;
