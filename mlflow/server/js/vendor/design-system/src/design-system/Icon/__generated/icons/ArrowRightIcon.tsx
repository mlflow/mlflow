import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgArrowRightIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="m15.06 8.03-7.03 7.03L6.97 14l5.22-5.22H1v-1.5h11.19L6.97 2.06 8.03 1z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ArrowRightIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgArrowRightIcon} />;
});
ArrowRightIcon.displayName = 'ArrowRightIcon';
export default ArrowRightIcon;
