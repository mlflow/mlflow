import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgBarsDescendingVerticalIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M7 12.75H1v-1.5h6zM15 4.75H1v-1.5h14zM1 7.25h10v1.5H1z" />
    </svg>
  );
}
const BarsDescendingVerticalIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgBarsDescendingVerticalIcon} />;
});
BarsDescendingVerticalIcon.displayName = 'BarsDescendingVerticalIcon';
export default BarsDescendingVerticalIcon;
