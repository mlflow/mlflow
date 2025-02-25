import type { Interpolation } from '@emotion/react';
import { Children } from 'react';

import type { Theme } from '../../../theme';
import { HintColumn } from '../../DropdownMenu/DropdownMenu';
import { InfoIcon } from '../../Icon';
import { LegacyTooltip } from '../../LegacyTooltip';

const infoIconStyles = (theme: Theme): Interpolation<Theme> => ({
  display: 'inline-flex',
  paddingLeft: theme.spacing.xs,
  color: theme.colors.textSecondary,
  pointerEvents: 'all',
});

export const getNewChildren = (
  children: React.ReactNode,
  props: { disabled?: boolean },
  disabledReason: React.ReactNode,
  ref: React.RefObject<HTMLElement>,
) => {
  const childCount = Children.count(children);
  const tooltip = (
    <LegacyTooltip
      title={disabledReason}
      placement="right"
      dangerouslySetAntdProps={{ getPopupContainer: () => ref.current || document.body }}
    >
      <span
        data-disabled-tooltip
        css={(theme) => infoIconStyles(theme)}
        onClick={(e) => {
          if (props.disabled) {
            e.stopPropagation();
          }
        }}
      >
        <InfoIcon role="presentation" alt="Disabled state reason" aria-hidden="false" />
      </span>
    </LegacyTooltip>
  );

  if (childCount === 1) {
    return getChild(children, Boolean(props['disabled']), disabledReason, tooltip, 0, childCount);
  }

  return Children.map(children, (child, idx) => {
    return getChild(child, Boolean(props['disabled']), disabledReason, tooltip, idx, childCount);
  });
};

const getChild = (
  child: React.ReactNode,
  isDisabled: boolean,
  disabledReason: React.ReactNode,
  tooltip: React.ReactElement,
  index: number,
  siblingCount: number,
) => {
  const HintColumnType = (<HintColumn />).type;
  const isHintColumnType = Boolean(
    child &&
      typeof child !== 'string' &&
      typeof child !== 'number' &&
      typeof child !== 'boolean' &&
      'type' in child &&
      child?.type === HintColumnType,
  );

  if (isDisabled && disabledReason && child && isHintColumnType) {
    return (
      <>
        {tooltip}
        {child}
      </>
    );
  } else if (index === siblingCount - 1 && isDisabled && disabledReason) {
    return (
      <>
        {child}
        {tooltip}
      </>
    );
  }
  return child;
};
