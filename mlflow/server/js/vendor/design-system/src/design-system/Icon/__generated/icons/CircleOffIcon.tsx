import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCircleOffIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="m11.667 5.392 1.362-1.363-1.06-1.06-9 9 1.06 1.06 1.363-1.362a4.5 4.5 0 0 0 6.276-6.276m-1.083 1.083-4.11 4.109a3 3 0 0 0 4.11-4.11M8 3.5q.606.002 1.164.152L7.811 5.006A3 3 0 0 0 5.006 7.81L3.652 9.164A4.5 4.5 0 0 1 8 3.5"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CircleOffIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCircleOffIcon} />;
});
CircleOffIcon.displayName = 'CircleOffIcon';
export default CircleOffIcon;
