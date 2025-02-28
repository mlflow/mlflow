import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgReplyIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <mask
        id="ReplyIcon_svg__a"
        width={16}
        height={16}
        x={0}
        y={0}
        maskUnits="userSpaceOnUse"
        style={{
          maskType: 'alpha',
        }}
      >
        <path fill="currentColor" d="M0 0h16v16H0z" />
      </mask>
      <g mask="url(#ReplyIcon_svg__a)">
        <path
          fill="currentColor"
          d="M3.333 3.333V6q0 .834.584 1.417Q4.5 8 5.333 8h6.117l-2.4-2.4.95-.933 4 4-4 4-.95-.934 2.4-2.4H5.333a3.21 3.21 0 0 1-2.358-.975A3.21 3.21 0 0 1 2 6V3.333z"
        />
      </g>
    </svg>
  );
}
const ReplyIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgReplyIcon} />;
});
ReplyIcon.displayName = 'ReplyIcon';
export default ReplyIcon;
