import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgSparkleDoubleFillIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M4.739 9.622a.75.75 0 0 0-1.478 0l-.152.876a.75.75 0 0 1-.61.61l-.878.153a.75.75 0 0 0 0 1.478l.877.152a.75.75 0 0 1 .61.61l.153.878a.75.75 0 0 0 1.478 0l.152-.877a.75.75 0 0 1 .61-.61l.878-.153a.75.75 0 0 0 0-1.478l-.877-.152a.75.75 0 0 1-.61-.61zM10.737.611a.75.75 0 0 0-1.474 0l-.264 1.398A3.75 3.75 0 0 1 6.01 5l-1.398.264a.75.75 0 0 0 0 1.474l1.398.264A3.75 3.75 0 0 1 9 9.99l.264 1.398a.75.75 0 0 0 1.474 0l.264-1.398A3.75 3.75 0 0 1 13.99 7l1.398-.264a.75.75 0 0 0 0-1.474l-1.398-.264A3.75 3.75 0 0 1 11 2.01z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const SparkleDoubleFillIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgSparkleDoubleFillIcon} />;
});
SparkleDoubleFillIcon.displayName = 'SparkleDoubleFillIcon';
export default SparkleDoubleFillIcon;
