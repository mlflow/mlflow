import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgLockUnlockedIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M10 11.75v-1.5H6v1.5z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M13.25 6H5.5V4a2.5 2.5 0 0 1 5 0v.5H12V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75M3.5 7.5h9v7h-9z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const LockUnlockedIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgLockUnlockedIcon} />;
});
LockUnlockedIcon.displayName = 'LockUnlockedIcon';
export default LockUnlockedIcon;
