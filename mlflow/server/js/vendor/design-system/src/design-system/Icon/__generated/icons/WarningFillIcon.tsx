import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgWarningFillIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8.649 1.374a.75.75 0 0 0-1.298 0l-7.25 12.5A.75.75 0 0 0 .75 15h14.5a.75.75 0 0 0 .649-1.126zM7.25 10V6.5h1.5V10zm1.5 1.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0"
        clipRule="evenodd"
      />
    </svg>
  );
}
const WarningFillIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgWarningFillIcon} />;
});
WarningFillIcon.displayName = 'WarningFillIcon';
export default WarningFillIcon;
