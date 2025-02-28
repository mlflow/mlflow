import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCaretDownSquareIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M8 10a.75.75 0 0 1-.59-.286l-2.164-2.75a.75.75 0 0 1 .589-1.214h4.33a.75.75 0 0 1 .59 1.214l-2.166 2.75A.75.75 0 0 1 8 10"
      />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CaretDownSquareIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCaretDownSquareIcon} />;
});
CaretDownSquareIcon.displayName = 'CaretDownSquareIcon';
export default CaretDownSquareIcon;
