import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgAlignLeftIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M1 2.5h14V1H1zM8 5.75H1v-1.5h7zM1 8.75v-1.5h14v1.5zM1 15v-1.5h14V15zM1 11.75h7v-1.5H1z"
      />
    </svg>
  );
}
const AlignLeftIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgAlignLeftIcon} />;
});
AlignLeftIcon.displayName = 'AlignLeftIcon';
export default AlignLeftIcon;
