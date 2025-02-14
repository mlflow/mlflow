import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgChevronUpIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8 7.083 5.053 10 4 8.958 8 5l4 3.958L10.947 10z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ChevronUpIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgChevronUpIcon} />;
});
ChevronUpIcon.displayName = 'ChevronUpIcon';
export default ChevronUpIcon;
