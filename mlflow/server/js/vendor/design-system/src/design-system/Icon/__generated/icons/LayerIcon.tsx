import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgLayerIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M13.5 2.5H7V1h7.25a.75.75 0 0 1 .75.75V9h-1.5z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1 7.75A.75.75 0 0 1 1.75 7h6.5a.75.75 0 0 1 .75.75v6.5a.75.75 0 0 1-.75.75h-6.5a.75.75 0 0 1-.75-.75zm1.5.75v5h5v-5z"
        clipRule="evenodd"
      />
      <path fill="currentColor" d="M4 5.32h6.5V12H12V4.57a.75.75 0 0 0-.75-.75H4z" />
    </svg>
  );
}
const LayerIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgLayerIcon} />;
});
LayerIcon.displayName = 'LayerIcon';
export default LayerIcon;
