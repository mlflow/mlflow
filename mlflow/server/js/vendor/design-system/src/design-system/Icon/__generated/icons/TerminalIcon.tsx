import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgTerminalIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M5.03 4.97 8.06 8l-3.03 3.03-1.06-1.06L5.94 8 3.97 6.03zM12 9.5H8V11h4z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v11h11v-11z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const TerminalIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgTerminalIcon} />;
});
TerminalIcon.displayName = 'TerminalIcon';
export default TerminalIcon;
