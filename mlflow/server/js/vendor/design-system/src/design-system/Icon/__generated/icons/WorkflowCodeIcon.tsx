import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgWorkflowCodeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <g fill="currentColor" clipPath="url(#WorkflowCodeIcon_svg__a)">
        <path d="m13.32 10.66-1.061 1.061 1.548 1.549-1.548 1.548 1.06 1.06 2.61-2.608zM10.68 10.66l1.061 1.061-1.548 1.549 1.548 1.548-1.06 1.06-2.61-2.608z" />
        <path
          fillRule="evenodd"
          d="M2.75 5.5c1.259 0 2.32-.846 2.646-2h5.229a1.875 1.875 0 0 1 .11 3.747l-3.3-2.357a.75.75 0 0 0-.87 0L3.256 7.252A3.375 3.375 0 0 0 3.375 14H7v-1.5H3.375a1.875 1.875 0 0 1-.11-3.747l3.3 2.357a.75.75 0 0 0 .87 0l3.308-2.362A3.375 3.375 0 0 0 10.625 2H5.396A2.751 2.751 0 1 0 2.75 5.5M4 2.75a1.25 1.25 0 1 1-2.5 0 1.25 1.25 0 0 1 2.5 0m3 3.672L4.79 8 7 9.578 9.21 8z"
          clipRule="evenodd"
        />
      </g>
      <defs>
        <clipPath>
          <path fill="#fff" d="M0 0h16v16H0z" />
        </clipPath>
      </defs>
    </svg>
  );
}
const WorkflowCodeIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgWorkflowCodeIcon} />;
});
WorkflowCodeIcon.displayName = 'WorkflowCodeIcon';
export default WorkflowCodeIcon;
