import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgPlayCircleFillIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m7.125-2.815A.75.75 0 0 0 6 5.835v4.33a.75.75 0 0 0 1.125.65l3.75-2.166a.75.75 0 0 0 0-1.299z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const PlayCircleFillIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgPlayCircleFillIcon} />;
});
PlayCircleFillIcon.displayName = 'PlayCircleFillIcon';
export default PlayCircleFillIcon;
