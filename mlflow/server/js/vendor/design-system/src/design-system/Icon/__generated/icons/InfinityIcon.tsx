import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgInfinityIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="m8 6.94 1.59-1.592a3.75 3.75 0 1 1 0 5.304L8 9.06l-1.591 1.59a3.75 3.75 0 1 1 0-5.303zm2.652-.531a2.25 2.25 0 1 1 0 3.182L9.06 8zM6.939 8 5.35 6.409a2.25 2.25 0 1 0 0 3.182l1.588-1.589z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const InfinityIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgInfinityIcon} />;
});
InfinityIcon.displayName = 'InfinityIcon';
export default InfinityIcon;
