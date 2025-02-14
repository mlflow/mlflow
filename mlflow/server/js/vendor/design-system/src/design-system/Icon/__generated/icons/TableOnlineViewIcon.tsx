import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgTableOnlineViewIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 15" {...props}>
      <path
        stroke="currentColor"
        strokeWidth={1.5}
        d="M.75 1A.25.25 0 0 1 1 .75h12a.25.25 0 0 1 .25.25v12a.25.25 0 0 1-.25.25H1A.25.25 0 0 1 .75 13zM1 5.25h12M5.25 13V6"
      />
      <path fill="#fff" d="M7 7h9v7.5H7z" />
      <path stroke="currentColor" d="M10.5 12H15m0 0-1.5-1.5M15 12l-1.5 1.5M12.5 9.5H8m0 0L9.5 11M8 9.5 9.5 8" />
    </svg>
  );
}
const TableOnlineViewIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgTableOnlineViewIcon} />;
});
TableOnlineViewIcon.displayName = 'TableOnlineViewIcon';
export default TableOnlineViewIcon;
