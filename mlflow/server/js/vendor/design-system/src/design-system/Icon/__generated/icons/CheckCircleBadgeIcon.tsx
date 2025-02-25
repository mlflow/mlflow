import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCheckCircleBadgeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="m10.47 5.47 1.06 1.06L7 11.06 4.47 8.53l1.06-1.06L7 8.94zM16 12.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0"
      />
      <path
        fill="currentColor"
        d="M1.5 8a6.5 6.5 0 0 1 13-.084c.54.236 1.031.565 1.452.967Q16 8.448 16 8a8 8 0 1 0-7.117 7.952 5 5 0 0 1-.967-1.453A6.5 6.5 0 0 1 1.5 8"
      />
    </svg>
  );
}
const CheckCircleBadgeIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCheckCircleBadgeIcon} />;
});
CheckCircleBadgeIcon.displayName = 'CheckCircleBadgeIcon';
export default CheckCircleBadgeIcon;
