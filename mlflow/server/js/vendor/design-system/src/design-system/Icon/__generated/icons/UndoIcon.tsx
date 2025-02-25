import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgUndoIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <g clipPath="url(#UndoIcon_svg__a)">
        <path
          fill="currentColor"
          d="M2.81 6.5h8.69a3 3 0 0 1 0 6H7V14h4.5a4.5 4.5 0 0 0 0-9H2.81l2.72-2.72-1.06-1.06-4.53 4.53 4.53 4.53 1.06-1.06z"
        />
      </g>
      <defs>
        <clipPath>
          <path fill="#fff" d="M16 16H0V0h16z" />
        </clipPath>
      </defs>
    </svg>
  );
}
const UndoIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgUndoIcon} />;
});
UndoIcon.displayName = 'UndoIcon';
export default UndoIcon;
