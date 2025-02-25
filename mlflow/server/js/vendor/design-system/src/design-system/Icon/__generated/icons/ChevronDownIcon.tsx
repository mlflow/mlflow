import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgChevronDownIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8 8.917 10.947 6 12 7.042 8 11 4 7.042 5.053 6z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ChevronDownIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgChevronDownIcon} />;
});
ChevronDownIcon.displayName = 'ChevronDownIcon';
export default ChevronDownIcon;
