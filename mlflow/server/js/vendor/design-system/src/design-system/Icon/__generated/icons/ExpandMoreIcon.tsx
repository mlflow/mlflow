import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgExpandMoreIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 17" {...props}>
      <path
        fill="currentColor"
        d="m4 4.03 1.06 1.061 2.97-2.97L11 5.091l1.06-1.06L8.03 0zM12.06 12.091l-4.03 4.03L4 12.091l1.06-1.06L8.03 14 11 11.03z"
      />
    </svg>
  );
}
const ExpandMoreIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgExpandMoreIcon} />;
});
ExpandMoreIcon.displayName = 'ExpandMoreIcon';
export default ExpandMoreIcon;
