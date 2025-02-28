import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgArrowInIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M4.5 2.5h9v11h-9V11H3v3.25c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H3.75a.75.75 0 0 0-.75.75V5h1.5z"
      />
      <path fill="currentColor" d="M12.06 8 8.03 3.97 6.97 5.03l2.22 2.22H1v1.5h8.19l-2.22 2.22 1.06 1.06z" />
    </svg>
  );
}
const ArrowInIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgArrowInIcon} />;
});
ArrowInIcon.displayName = 'ArrowInIcon';
export default ArrowInIcon;
