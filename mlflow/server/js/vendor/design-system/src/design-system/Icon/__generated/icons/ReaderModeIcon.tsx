import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgReaderModeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="M13 4.5h-3V6h3zM13 7.25h-3v1.5h3zM13 10h-3v1.5h3z" />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 10.5v-9h5.75v9zm7.25 0h5.75v-9H8.75z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const ReaderModeIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgReaderModeIcon} />;
});
ReaderModeIcon.displayName = 'ReaderModeIcon';
export default ReaderModeIcon;
