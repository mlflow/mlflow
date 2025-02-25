import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgPanelFloatingIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11zM5.6 5a.6.6 0 0 0-.6.6v4.8a.6.6 0 0 0 .6.6h4.8a.6.6 0 0 0 .6-.6V5.6a.6.6 0 0 0-.6-.6z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const PanelFloatingIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgPanelFloatingIcon} />;
});
PanelFloatingIcon.displayName = 'PanelFloatingIcon';
export default PanelFloatingIcon;
