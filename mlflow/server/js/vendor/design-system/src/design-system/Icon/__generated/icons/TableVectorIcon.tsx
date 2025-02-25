import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgTableVectorIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H8v-1.5H6.5V7h7v2H15V1.75a.75.75 0 0 0-.75-.75zM5 7H2.5v6.5H5zm8.5-1.5v-3h-11v3z"
        clipRule="evenodd"
      />
      <circle cx={12} cy={12} r={1} fill="currentColor" />
      <circle cx={12.5} cy={9.5} r={0.5} fill="currentColor" />
      <circle cx={9.5} cy={12.5} r={0.5} fill="currentColor" />
      <circle cx={13.75} cy={13.75} r={0.75} fill="currentColor" />
      <circle cx={9.75} cy={9.75} r={0.75} fill="currentColor" />
      <path stroke="currentColor" strokeWidth={0.3} d="M13.5 13.5 12 12m0 0 .5-2.5M12 12l-2.5.5M10 10l2 2" />
    </svg>
  );
}
const TableVectorIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgTableVectorIcon} />;
});
TableVectorIcon.displayName = 'TableVectorIcon';
export default TableVectorIcon;
