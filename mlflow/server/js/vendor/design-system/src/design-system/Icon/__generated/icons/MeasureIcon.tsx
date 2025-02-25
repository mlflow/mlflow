import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgMeasureIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="m14.884 4.409-3.293-3.293a1.25 1.25 0 0 0-1.768 0L1.116 9.823a1.25 1.25 0 0 0 0 1.768l3.293 3.293a1.25 1.25 0 0 0 1.768 0l8.707-8.707a1.25 1.25 0 0 0 0-1.768m-9.592 9.237L2.355 10.71 4 9.063l1.47 1.47A.751.751 0 1 0 6.532 9.47L5.062 8 6 7.063l1.47 1.47A.751.751 0 0 0 8.531 7.47L7.062 6 8 5.063l1.47 1.47a.751.751 0 1 0 1.062-1.063L9.062 4l1.647-1.646 2.938 2.937z"
      />
    </svg>
  );
}
const MeasureIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgMeasureIcon} />;
});
MeasureIcon.displayName = 'MeasureIcon';
export default MeasureIcon;
