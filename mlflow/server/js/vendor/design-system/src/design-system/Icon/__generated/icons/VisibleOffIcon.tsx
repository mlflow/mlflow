import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgVisibleOffIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <g fill="currentColor" clipPath="url(#VisibleOffIcon_svg__a)">
        <path
          fillRule="evenodd"
          d="m11.634 13.195 1.335 1.335 1.061-1.06-11.5-11.5-1.06 1.06 1.027 1.028a8.4 8.4 0 0 0-2.469 3.72.75.75 0 0 0 0 .465 8.39 8.39 0 0 0 11.606 4.951m-1.14-1.14-1.301-1.301a3 3 0 0 1-3.946-3.946L3.56 5.121A6.9 6.9 0 0 0 1.535 8.01a6.89 6.89 0 0 0 8.96 4.045"
          clipRule="evenodd"
        />
        <path d="M15.972 8.243a8.4 8.4 0 0 1-1.946 3.223l-1.06-1.06a6.9 6.9 0 0 0 1.499-2.396 6.89 6.89 0 0 0-8.187-4.293L5.082 2.522a8.389 8.389 0 0 1 10.89 5.256.75.75 0 0 1 0 .465" />
        <path d="M11 8q0 .21-.028.411L7.589 5.028q.201-.027.41-.028a3 3 0 0 1 3 3" />
      </g>
      <defs>
        <clipPath>
          <path fill="#fff" d="M0 0h16v16H0z" />
        </clipPath>
      </defs>
    </svg>
  );
}
const VisibleOffIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgVisibleOffIcon} />;
});
VisibleOffIcon.displayName = 'VisibleOffIcon';
export default VisibleOffIcon;
