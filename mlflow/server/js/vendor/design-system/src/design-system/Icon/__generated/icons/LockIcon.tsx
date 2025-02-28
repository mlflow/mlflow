import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgLockIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M7.25 9v4h1.5V9z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M12 6V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75zm.5 1.5v7h-9v-7zM5.5 4v2h5V4a2.5 2.5 0 0 0-5 0"
        clipRule="evenodd"
      />
    </svg>
  );
}
const LockIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgLockIcon} />;
});
LockIcon.displayName = 'LockIcon';
export default LockIcon;
