import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgBoldIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M4.75 3a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h4.375a2.875 2.875 0 0 0 1.496-5.33A2.875 2.875 0 0 0 8.375 3zm.75 5.75v2.75h3.625a1.375 1.375 0 0 0 0-2.75zm2.877-1.5a1.375 1.375 0 0 0-.002-2.75H5.5v2.75z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const BoldIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgBoldIcon} />;
});
BoldIcon.displayName = 'BoldIcon';
export default BoldIcon;
