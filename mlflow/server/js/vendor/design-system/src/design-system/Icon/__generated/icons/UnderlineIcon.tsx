import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgUnderlineIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M4.544 6.466 4.6 2.988l1.5.024-.056 3.478A1.978 1.978 0 1 0 10 6.522V3h1.5v3.522a3.478 3.478 0 1 1-6.956-.056M12 13H4v-1.5h8z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const UnderlineIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgUnderlineIcon} />;
});
UnderlineIcon.displayName = 'UnderlineIcon';
export default UnderlineIcon;
