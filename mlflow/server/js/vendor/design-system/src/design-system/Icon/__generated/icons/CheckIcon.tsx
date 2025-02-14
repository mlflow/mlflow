import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCheckIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="m15.06 3.56-9.53 9.531L1 8.561 2.06 7.5l3.47 3.47L14 2.5z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CheckIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCheckIcon} />;
});
CheckIcon.displayName = 'CheckIcon';
export default CheckIcon;
