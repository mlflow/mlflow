import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgZoomOutIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 17" {...props}>
      <path fill="currentColor" d="M11 7.25H5v1.5h6z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1 8a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 1 8m7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ZoomOutIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgZoomOutIcon} />;
});
ZoomOutIcon.displayName = 'ZoomOutIcon';
export default ZoomOutIcon;
