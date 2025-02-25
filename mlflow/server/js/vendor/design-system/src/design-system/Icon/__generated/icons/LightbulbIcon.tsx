import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgLightbulbIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path stroke="currentColor" strokeLinecap="round" strokeWidth={1.5} d="M7 15h2" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M9.475 10.019a2.5 2.5 0 1 0-2.95 0c.528.386.975 1.048.975 1.879V12h1v-.102c0-.83.447-1.492.975-1.88m.887 1.21a.84.84 0 0 0-.362.669v.852a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1-.75-.75v-.852a.84.84 0 0 0-.362-.67 4 4 0 1 1 4.724 0"
        clipRule="evenodd"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth={1.5}
        d="M11.375 3.78 12 3M13.051 6.316 14 6M4.625 3.78 4 3M2.949 6.316 2 6M8 2.5v-1"
      />
    </svg>
  );
}
const LightbulbIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgLightbulbIcon} />;
});
LightbulbIcon.displayName = 'LightbulbIcon';
export default LightbulbIcon;
