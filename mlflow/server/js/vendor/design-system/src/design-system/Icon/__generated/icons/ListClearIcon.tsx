import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgListClearIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <g fill="currentColor" clipPath="url(#ListClearIcon_svg__a)">
        <path d="M15.03 13.97 13.06 12l1.97-1.97-1.06-1.06L12 10.94l-1.97-1.97-1.06 1.06L10.94 12l-1.97 1.97 1.06 1.06L12 13.06l1.97 1.97zM5 11.5H1V10h4zM11 3.5H1V2h10zM7 7.5H1V6h6z" />
      </g>
      <defs>
        <clipPath>
          <path fill="#fff" d="M0 16h16V0H0z" />
        </clipPath>
      </defs>
    </svg>
  );
}
const ListClearIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgListClearIcon} />;
});
ListClearIcon.displayName = 'ListClearIcon';
export default ListClearIcon;
