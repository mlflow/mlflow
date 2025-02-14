import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgPauseIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" fillRule="evenodd" d="M10 12V4h1.5v8zm-5.5 0V4H6v8z" clipRule="evenodd" />
    </svg>
  );
}
const PauseIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgPauseIcon} />;
});
PauseIcon.displayName = 'PauseIcon';
export default PauseIcon;
