import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCodeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 17 16" {...props}>
      <path
        fill="currentColor"
        d="M4.03 12.06 5.091 11l-2.97-2.97 2.97-2.97L4.031 4 0 8.03zM12.091 4l4.03 4.03-4.03 4.03-1.06-1.06L14 8.03l-2.97-2.97z"
      />
    </svg>
  );
}
const CodeIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCodeIcon} />;
});
CodeIcon.displayName = 'CodeIcon';
export default CodeIcon;
