import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgDragIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M5.25 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M5.25 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M5.25 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5"
      />
    </svg>
  );
}
const DragIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgDragIcon} />;
});
DragIcon.displayName = 'DragIcon';
export default DragIcon;
