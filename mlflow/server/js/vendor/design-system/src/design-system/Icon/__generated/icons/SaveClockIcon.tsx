import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgSaveClockIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8m-.75-6.5v2.81l1.72 1.72 1.06-1.06-1.28-1.28V9.5z"
        clipRule="evenodd"
      />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h5.941a5.2 5.2 0 0 1-.724-1.5H2.5v-11H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75V2.81l2.5 2.5v1.657a5.2 5.2 0 0 1 1.5.724V5a.75.75 0 0 0-.22-.53l-3.25-3.25A.75.75 0 0 0 11 1zM6.5 2.5h3v3h-3z"
        clipRule="evenodd"
      />
      <path fill="currentColor" d="M7.527 9.25H5v1.5h1.9a5.2 5.2 0 0 1 .627-1.5" />
    </svg>
  );
}
const SaveClockIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgSaveClockIcon} />;
});
SaveClockIcon.displayName = 'SaveClockIcon';
export default SaveClockIcon;
