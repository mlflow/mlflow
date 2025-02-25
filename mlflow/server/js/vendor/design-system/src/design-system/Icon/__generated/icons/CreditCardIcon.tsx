import * as React from 'react';
import { forwardRef } from 'react';
import type { Ref } from 'react';

import type { IconProps } from '../../Icon';
import { Icon } from '../../Icon';
function SvgCreditCardIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        d="M14.4 4.4v7.2q0 .483-.353.842a1.15 1.15 0 0 1-.847.358H2.8q-.483 0-.842-.358A1.15 1.15 0 0 1 1.6 11.6V4.4q0-.483.358-.841.36-.36.842-.359h10.4q.495 0 .848.359.352.357.352.841M2.8 5.6h10.4V4.4H2.8zm0 2.4v3.6h10.4V8z"
      />
    </svg>
  );
}
const CreditCardIcon = forwardRef((props: IconProps, forwardedRef?: Ref<HTMLSpanElement>) => {
  return <Icon ref={forwardedRef} {...props} component={SvgCreditCardIcon} />;
});
CreditCardIcon.displayName = 'CreditCardIcon';
export default CreditCardIcon;
