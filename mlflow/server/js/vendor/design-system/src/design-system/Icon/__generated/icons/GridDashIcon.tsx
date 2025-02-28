import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgGridDashIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 0 0-.75.75M15 14.25V12h-1.5v1.5H12V15h2.25a.75.75 0 0 0 .75-.75M12 1h2.25a.75.75 0 0 1 .75.75V4h-1.5V2.5H12zM1.75 15H4v-1.5H2.5V12H1v2.25a.75.75 0 0 0 .75.75M10 2.5H6V1h4zM6 15h4v-1.5H6zM13.5 10V6H15v4zM1 6v4h1.5V6z"
      />
    </svg>
  );
}
const GridDashIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgGridDashIcon} />;
});
GridDashIcon.displayName = 'GridDashIcon';
export default GridDashIcon;
