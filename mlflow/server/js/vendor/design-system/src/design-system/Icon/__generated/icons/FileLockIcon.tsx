import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgFileLockIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <g fill="currentColor" fillRule="evenodd" clipPath="url(#FileLockIcon_svg__a)" clipRule="evenodd">
        <path d="M2.75 0A.75.75 0 0 0 2 .75v13.5c0 .414.336.75.75.75H7.5v-1.5h-4v-12H8v3.75c0 .414.336.75.75.75h3.75v1H14V5.25a.75.75 0 0 0-.22-.53L9.28.22A.75.75 0 0 0 8.75 0zm8.69 4.5L9.5 2.56V4.5z" />
        <path d="M14 10v.688h.282a.75.75 0 0 1 .75.75v3.874a.75.75 0 0 1-.75.75H9.718a.75.75 0 0 1-.75-.75v-3.874a.75.75 0 0 1 .75-.75H10V10a2 2 0 0 1 4 0m-1.5 0v.688h-1V10a.5.5 0 0 1 1 0m1.032 2.188v2.374h-3.064v-2.374z" />
      </g>
      <defs>
        <clipPath>
          <path fill="#fff" d="M0 0h16v16H0z" />
        </clipPath>
      </defs>
    </svg>
  );
}
const FileLockIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgFileLockIcon} />;
});
FileLockIcon.displayName = 'FileLockIcon';
export default FileLockIcon;
