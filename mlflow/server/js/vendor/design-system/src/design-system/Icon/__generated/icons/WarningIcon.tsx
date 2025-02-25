import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgWarningIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M7.25 10V6.5h1.5V10zM8 12.5A.75.75 0 1 0 8 11a.75.75 0 0 0 0 1.5" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8 1a.75.75 0 0 1 .649.374l7.25 12.5A.75.75 0 0 1 15.25 15H.75a.75.75 0 0 1-.649-1.126l7.25-12.5A.75.75 0 0 1 8 1m0 2.245L2.052 13.5h11.896z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const WarningIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgWarningIcon} />;
});
WarningIcon.displayName = 'WarningIcon';
export default WarningIcon;
