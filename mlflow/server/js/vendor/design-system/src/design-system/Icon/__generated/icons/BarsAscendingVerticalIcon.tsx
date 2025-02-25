import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgBarsAscendingVerticalIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M7 3.25H1v1.5h6zM15 11.25H1v1.5h14zM1 8.75h10v-1.5H1z" />
    </svg>
  );
}
const BarsAscendingVerticalIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgBarsAscendingVerticalIcon} />;
});
BarsAscendingVerticalIcon.displayName = 'BarsAscendingVerticalIcon';
export default BarsAscendingVerticalIcon;
