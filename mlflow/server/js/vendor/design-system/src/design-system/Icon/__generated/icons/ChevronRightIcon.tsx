import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgChevronRightIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8.917 8 6 5.053 7.042 4 11 8l-3.958 4L6 10.947z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ChevronRightIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgChevronRightIcon} />;
});
ChevronRightIcon.displayName = 'ChevronRightIcon';
export default ChevronRightIcon;
