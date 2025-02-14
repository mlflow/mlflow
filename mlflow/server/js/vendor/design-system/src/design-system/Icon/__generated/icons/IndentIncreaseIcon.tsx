import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgIndentIncreaseIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M16 2H0v1.5h16zM16 5.5H8V7h8zM16 9H8v1.5h8zM0 12.5V14h16v-1.5zM1.97 6.03l1.06-1.06L6.06 8l-3.03 3.03-1.06-1.06L3.94 8z"
      />
    </svg>
  );
}
const IndentIncreaseIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgIndentIncreaseIcon} />;
});
IndentIncreaseIcon.displayName = 'IndentIncreaseIcon';
export default IndentIncreaseIcon;
