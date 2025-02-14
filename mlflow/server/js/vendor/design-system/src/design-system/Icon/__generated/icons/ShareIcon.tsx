import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgShareIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M3.97 5.03 8 1l4.03 4.03-1.06 1.061-2.22-2.22v7.19h-1.5V3.87l-2.22 2.22z" />
      <path
        fill="currentColor"
        d="M2.5 13.56v-6.5H1v7.25c0 .415.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V7.06h-1.5v6.5z"
      />
    </svg>
  );
}
const ShareIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgShareIcon} />;
});
ShareIcon.displayName = 'ShareIcon';
export default ShareIcon;
