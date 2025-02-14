import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCheckCircleIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M11.53 6.53 7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CheckCircleIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCheckCircleIcon} />;
});
CheckCircleIcon.displayName = 'CheckCircleIcon';
export default CheckCircleIcon;
