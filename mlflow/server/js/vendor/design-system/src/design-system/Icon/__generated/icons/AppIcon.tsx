import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgAppIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M2.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M2.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M2.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5"
        clipRule="evenodd"
      />
    </svg>
  );
}
const AppIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgAppIcon} />;
});
AppIcon.displayName = 'AppIcon';
export default AppIcon;
