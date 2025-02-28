import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgRedoIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <g clipPath="url(#RedoIcon_svg__a)">
        <path
          fill="currentColor"
          fillRule="evenodd"
          d="m13.19 5-2.72-2.72 1.06-1.06 4.53 4.53-4.53 4.53-1.06-1.06 2.72-2.72H4.5a3 3 0 1 0 0 6H9V14H4.5a4.5 4.5 0 0 1 0-9z"
          clipRule="evenodd"
        />
      </g>
      <defs>
        <clipPath>
          <path fill="#fff" d="M0 16h16V0H0z" />
        </clipPath>
      </defs>
    </svg>
  );
}
const RedoIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgRedoIcon} />;
});
RedoIcon.displayName = 'RedoIcon';
export default RedoIcon;
