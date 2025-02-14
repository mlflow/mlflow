import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgExpandLessIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 17" {...props}>
      <path
        fill="currentColor"
        d="M12.06 1.06 11 0 8.03 2.97 5.06 0 4 1.06l4.03 4.031zM4 15l4.03-4.03L12.06 15 11 16.06l-2.97-2.969-2.97 2.97z"
      />
    </svg>
  );
}
const ExpandLessIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgExpandLessIcon} />;
});
ExpandLessIcon.displayName = 'ExpandLessIcon';
export default ExpandLessIcon;
