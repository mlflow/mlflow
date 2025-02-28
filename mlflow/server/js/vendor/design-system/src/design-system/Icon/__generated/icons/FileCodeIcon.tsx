import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgFileCodeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 17" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z"
        clipRule="evenodd"
      />
      <path
        fill="currentColor"
        d="M7.47 9.97 4.44 13l3.03 3.03 1.06-1.06L6.56 13l1.97-1.97zM11.03 9.97l-1.06 1.06L11.94 13l-1.97 1.97 1.06 1.06L14.06 13z"
      />
    </svg>
  );
}
const FileCodeIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgFileCodeIcon} />;
});
FileCodeIcon.displayName = 'FileCodeIcon';
export default FileCodeIcon;
