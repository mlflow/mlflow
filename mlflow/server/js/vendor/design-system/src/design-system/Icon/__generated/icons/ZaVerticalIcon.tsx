import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgZaVerticalIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M7.996 8a.75.75 0 0 0-.695.468L4.65 15h1.619l.406-1h2.643l.406 1h1.619L8.69 8.468A.75.75 0 0 0 7.996 8m.713 4.5H7.284l.712-1.756zM8.664 1.5H4.996V0h5.25a.75.75 0 0 1 .58 1.225L7.33 5.5h3.667V7h-5.25a.75.75 0 0 1-.58-1.225z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ZaVerticalIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgZaVerticalIcon} />;
});
ZaVerticalIcon.displayName = 'ZaVerticalIcon';
export default ZaVerticalIcon;
