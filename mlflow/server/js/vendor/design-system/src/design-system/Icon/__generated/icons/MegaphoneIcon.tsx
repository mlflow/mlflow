import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgMegaphoneIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 18 18" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M16.25 2a.75.75 0 0 0-1.248-.56l-4.287 3.81H4A2.75 2.75 0 0 0 1.25 8v2A2.75 2.75 0 0 0 4 12.75h1.75V16a.75.75 0 0 0 1.5 0v-3.25h3.465l4.287 3.81A.75.75 0 0 0 16.25 16zm-4.752 4.56 3.252-2.89v10.66l-3.252-2.89a.75.75 0 0 0-.498-.19H4c-.69 0-1.25-.56-1.25-1.25V8c0-.69.56-1.25 1.25-1.25h7a.75.75 0 0 0 .498-.19"
        clipRule="evenodd"
      />
    </svg>
  );
}
const MegaphoneIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgMegaphoneIcon} />;
});
MegaphoneIcon.displayName = 'MegaphoneIcon';
export default MegaphoneIcon;
