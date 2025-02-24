import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgNewWindowIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M10 1h5v5h-1.5V3.56L8.53 8.53 7.47 7.47l4.97-4.97H10z" />
      <path
        fill="currentColor"
        d="M1 2.75A.75.75 0 0 1 1.75 2H8v1.5H2.5v10h10V8H14v6.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75z"
      />
    </svg>
  );
}
const NewWindowIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgNewWindowIcon} />;
});
NewWindowIcon.displayName = 'NewWindowIcon';
export default NewWindowIcon;
