import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgLibrariesIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path fill="currentColor" d="m8.301 1.522 5.25 13.5 1.398-.544-5.25-13.5zM1 15V1h1.5v14zM5 15V1h1.5v14z" />
    </svg>
  );
}
const LibrariesIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgLibrariesIcon} />;
});
LibrariesIcon.displayName = 'LibrariesIcon';
export default LibrariesIcon;
