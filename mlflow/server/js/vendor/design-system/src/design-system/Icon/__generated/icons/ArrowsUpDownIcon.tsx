import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgArrowsUpDownIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M5.03 1 1 5.03l1.06 1.061 2.22-2.22v6.19h1.5V3.87L8 6.091l1.06-1.06zM11.03 15.121l4.03-4.03-1.06-1.06-2.22 2.219V6.06h-1.5v6.19l-2.22-2.22L7 11.091z"
      />
    </svg>
  );
}
const ArrowsUpDownIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgArrowsUpDownIcon} />;
});
ArrowsUpDownIcon.displayName = 'ArrowsUpDownIcon';
export default ArrowsUpDownIcon;
