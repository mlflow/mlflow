import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCloseIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M6.97 8.03 2 3.06 3.06 2l4.97 4.97L13 2l1.06 1.06-4.969 4.97 4.97 4.97L13 14.06 8.03 9.092l-4.97 4.97L2 13z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CloseIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCloseIcon} />;
});
CloseIcon.displayName = 'CloseIcon';
export default CloseIcon;
