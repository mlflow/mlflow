import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgFaceNeutralIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <g fill="currentColor" fillRule="evenodd" clipPath="url(#FaceNeutralIcon_svg__a)" clipRule="evenodd">
        <path d="M8 2.084a5.917 5.917 0 1 0 0 11.833A5.917 5.917 0 0 0 8 2.084M.583 8a7.417 7.417 0 1 1 14.834 0A7.417 7.417 0 0 1 .583 8" />
        <path d="M4.583 10a.75.75 0 0 1 .75-.75h5.334a.75.75 0 1 1 0 1.5H5.333a.75.75 0 0 1-.75-.75M5.25 6A.75.75 0 0 1 6 5.25h.007a.75.75 0 0 1 0 1.5H6A.75.75 0 0 1 5.25 6M9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 1 1 0 1.5H10A.75.75 0 0 1 9.25 6" />
      </g>
      <defs>
        <clipPath>
          <path fill="#fff" d="M0 0h16v16H0z" />
        </clipPath>
      </defs>
    </svg>
  );
}
const FaceNeutralIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgFaceNeutralIcon} />;
});
FaceNeutralIcon.displayName = 'FaceNeutralIcon';
export default FaceNeutralIcon;
