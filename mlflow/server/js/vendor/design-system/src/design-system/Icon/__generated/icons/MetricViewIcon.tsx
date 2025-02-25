import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgMetricViewIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H4v-1.5H2.5V7H5v2h1.5V7h3v2H11V7h2.5v2H15V1.75a.75.75 0 0 0-.75-.75zM13.5 5.5v-3h-11v3z"
        clipRule="evenodd"
      />
      <path
        fill="currentColor"
        d="M11.25 15v-3.45a.75.75 0 0 0-1.5 0V15zM13.626 15v-2.65a.75.75 0 0 0-1.5 0V15zM8.874 15v-2.65a.75.75 0 0 0-1.5 0V15zM15.202 15a.8.8 0 0 0 .8-.8v-2.65a.75.75 0 0 0-1.5 0V15zM6.498 15v-3.45c0-.414-.288-.75-.75-.75-.366 0-.702.336-.75.75v2.65a.8.8 0 0 0 .8.8z"
      />
      <path
        fill="currentColor"
        d="M5.22 14.78c.14.141.331.22.53.22h9.5a.75.75 0 0 0 .75-.75V12.5h-1.5v1h-8v-1H5v1.75c0 .199.079.39.22.53"
      />
    </svg>
  );
}
const MetricViewIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgMetricViewIcon} />;
});
MetricViewIcon.displayName = 'MetricViewIcon';
export default MetricViewIcon;
