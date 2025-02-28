import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgSaveIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M10 9.25H6v1.5h4z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1 1.75A.75.75 0 0 1 1.75 1H11a.75.75 0 0 1 .53.22l3.25 3.25c.141.14.22.331.22.53v9.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75V2.81l2.5 2.5v8.19h-11zm4 0h3v3h-3z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const SaveIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgSaveIcon} />;
});
SaveIcon.displayName = 'SaveIcon';
export default SaveIcon;
