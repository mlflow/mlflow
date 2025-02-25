import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgBarsDescendingHorizontalIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M12.75 9v6h-1.5V9zM4.75 1v14h-1.5V1zM7.25 15V5h1.5v10z" />
    </svg>
  );
}
const BarsDescendingHorizontalIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgBarsDescendingHorizontalIcon} />;
});
BarsDescendingHorizontalIcon.displayName = 'BarsDescendingHorizontalIcon';
export default BarsDescendingHorizontalIcon;
