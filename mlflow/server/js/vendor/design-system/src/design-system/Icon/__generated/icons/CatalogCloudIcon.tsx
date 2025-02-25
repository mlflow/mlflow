import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCatalogCloudIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M2.5 13.25V4.792c.306.134.644.208 1 .208h8v1H13V.75a.75.75 0 0 0-.75-.75H3.5A2.5 2.5 0 0 0 1 2.5v10.75A2.75 2.75 0 0 0 3.75 16H4v-1.5h-.25c-.69 0-1.25-.56-1.25-1.25m9-9.75h-8a1 1 0 0 1 0-2h8z"
        clipRule="evenodd"
      />
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M10.179 7a3.61 3.61 0 0 0-3.464 2.595 3.251 3.251 0 0 0 .443 6.387.8.8 0 0 0 .163.018h5.821C14.758 16 16 14.688 16 13.107c0-1.368-.931-2.535-2.229-2.824A3.61 3.61 0 0 0 10.18 7m-2.805 7.496q.023 0 .044.004h5.555a1 1 0 0 1 .1-.002l.07.002c.753 0 1.357-.607 1.357-1.393s-.604-1.393-1.357-1.393h-.107a.75.75 0 0 1-.75-.75v-.357a2.107 2.107 0 0 0-4.199-.26.75.75 0 0 1-.698.656 1.75 1.75 0 0 0-.015 3.493"
        clipRule="evenodd"
      />
    </svg>
  );
}
const CatalogCloudIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCatalogCloudIcon} />;
});
CatalogCloudIcon.displayName = 'CatalogCloudIcon';
export default CatalogCloudIcon;
