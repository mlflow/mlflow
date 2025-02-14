import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgAlignCenterIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M1 2.5h14V1H1zM11.5 5.75h-7v-1.5h7zM15 8.75H1v-1.5h14zM15 15H1v-1.5h14zM4.5 11.75h7v-1.5h-7z"
      />
    </svg>
  );
}
const AlignCenterIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgAlignCenterIcon} />;
});
AlignCenterIcon.displayName = 'AlignCenterIcon';
export default AlignCenterIcon;
