import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgArrowDownIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8.03 15.06 1 8.03l1.06-1.06 5.22 5.22V1h1.5v11.19L14 6.97l1.06 1.06z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ArrowDownIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgArrowDownIcon} />;
});
ArrowDownIcon.displayName = 'ArrowDownIcon';
export default ArrowDownIcon;
