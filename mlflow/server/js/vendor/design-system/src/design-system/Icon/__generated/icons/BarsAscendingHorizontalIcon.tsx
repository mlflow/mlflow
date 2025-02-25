import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgBarsAscendingHorizontalIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M3.25 9v6h1.5V9zM11.25 1v14h1.5V1zM8.75 15V5h-1.5v10z" />
    </svg>
  );
}
const BarsAscendingHorizontalIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgBarsAscendingHorizontalIcon} />;
});
BarsAscendingHorizontalIcon.displayName = 'BarsAscendingHorizontalIcon';
export default BarsAscendingHorizontalIcon;
