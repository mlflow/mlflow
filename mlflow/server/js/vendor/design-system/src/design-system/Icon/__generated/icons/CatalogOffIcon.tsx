import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCatalogOffIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M14.03.75v10.69l-1.5-1.5V1.5H4.78c-.2 0-.39.047-.558.131L3.136.545A2.74 2.74 0 0 1 4.78 0h8.5a.75.75 0 0 1 .75.75"
      />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M2.03 3.56 1 2.53l1.06-1.06 13 13L14 15.53l-.017-.017a.75.75 0 0 1-.703.487H4.53a2.5 2.5 0 0 1-2.5-2.5zm8.94 8.94 1.56 1.56v.44h-8a1 1 0 1 1 0-2zM9.47 11H4.53c-.355 0-.693.074-1 .208V5.061z"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CatalogOffIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCatalogOffIcon} />;
});
CatalogOffIcon.displayName = 'CatalogOffIcon';
export default CatalogOffIcon;
