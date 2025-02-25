import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCheckLineIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M15.06 2.06 14 1 5.53 9.47 2.06 6 1 7.06l4.53 4.531zM1.03 15.03h14v-1.5h-14z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CheckLineIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCheckLineIcon} />;
});
CheckLineIcon.displayName = 'CheckLineIcon';
export default CheckLineIcon;
