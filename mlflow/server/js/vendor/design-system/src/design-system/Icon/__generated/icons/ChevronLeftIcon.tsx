import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgChevronLeftIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M7.083 8 10 10.947 8.958 12 5 8l3.958-4L10 5.053z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ChevronLeftIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgChevronLeftIcon} />;
});
ChevronLeftIcon.displayName = 'ChevronLeftIcon';
export default ChevronLeftIcon;
