import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgGitCommitIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8 5.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5M4.07 7.25a4.001 4.001 0 0 1 7.86 0H16v1.5h-4.07a4.001 4.001 0 0 1-7.86 0H0v-1.5z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const GitCommitIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgGitCommitIcon} />;
});
GitCommitIcon.displayName = 'GitCommitIcon';
export default GitCommitIcon;
