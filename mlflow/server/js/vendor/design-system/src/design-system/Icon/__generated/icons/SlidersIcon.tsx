import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgSlidersIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M2 3.104V2h1.5v1.104a2.751 2.751 0 0 1 0 5.292V14H2V8.396a2.751 2.751 0 0 1 0-5.292M1.5 5.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0M12.5 2v1.104a2.751 2.751 0 0 0 0 5.292V14H14V8.396a2.751 2.751 0 0 0 0-5.292V2zm.75 2.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M7.25 14v-1.104a2.751 2.751 0 0 1 0-5.292V2h1.5v5.604a2.751 2.751 0 0 1 0 5.292V14zM8 11.5A1.25 1.25 0 1 1 8 9a1.25 1.25 0 0 1 0 2.5"
        clipRule="evenodd"
      />
    </svg>
  );
}
const SlidersIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgSlidersIcon} />;
});
SlidersIcon.displayName = 'SlidersIcon';
export default SlidersIcon;
