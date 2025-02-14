import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgLeafIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M5 6a2.75 2.75 0 0 0 2.75 2.75h2.395a2 2 0 1 0 0-1.5H7.75C7.06 7.25 6.5 6.69 6.5 6V2H5zm6.5 2a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0"
        clipRule="evenodd"
      />
    </svg>
  );
}
const LeafIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgLeafIcon} />;
});
LeafIcon.displayName = 'LeafIcon';
export default LeafIcon;
