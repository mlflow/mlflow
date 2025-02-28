import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgBarGroupedIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M12.25 2a.75.75 0 0 0-.75.75V7H9.25a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm-.75 10.5v-4H10v4zm1.5 0h1.5v-9H13zM3.75 5a.75.75 0 0 0-.75.75V9H.75a.75.75 0 0 0-.75.75v3.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75v-7.5A.75.75 0 0 0 6.75 5zM3 12.5v-2H1.5v2zm1.5 0H6v-6H4.5z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const BarGroupedIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgBarGroupedIcon} />;
});
BarGroupedIcon.displayName = 'BarGroupedIcon';
export default BarGroupedIcon;
