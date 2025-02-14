import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgZaHorizontalIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M11.654 4.5a.75.75 0 0 1 .695.468L15 11.5h-1.619l-.406-1h-2.643l-.406 1H8.307l2.652-6.532a.75.75 0 0 1 .695-.468M10.94 9h1.425l-.712-1.756zM4.667 6H1V4.5h5.25a.75.75 0 0 1 .58 1.225L3.333 10H7v1.5H1.75a.75.75 0 0 1-.58-1.225z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ZaHorizontalIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgZaHorizontalIcon} />;
});
ZaHorizontalIcon.displayName = 'ZaHorizontalIcon';
export default ZaHorizontalIcon;
