import type { Interpolation, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { Children } from 'react';

import { OverflowPopover } from './OverflowPopover';
import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import type { PopoverProps } from '../Popover/Popover';
import { Tag } from '../Tag';
export interface OverflowProps extends PopoverProps {
  /** Used for components like Tag which have already have margins */
  noMargin?: boolean;
  children: React.ReactNode;
}

export const Overflow = ({ children, noMargin = false, ...props }: OverflowProps): React.ReactElement => {
  const { theme } = useDesignSystemTheme();

  const childrenList = children && Children.toArray(children);

  if (!childrenList || childrenList.length === 0) {
    return <>{children}</>;
  }

  const firstItem = childrenList[0];
  const additionalItems = childrenList.splice(1);

  const renderOverflowLabel = (label: string) => (
    <Tag componentId="codegen_design-system_src_design-system_overflow_overflow.tsx_28" css={getTagStyles(theme)}>
      {label}
    </Tag>
  );

  return additionalItems.length === 0 ? (
    <>{firstItem}</>
  ) : (
    <div
      {...props}
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: noMargin ? 0 : theme.spacing.sm,
        maxWidth: '100%',
      }}
    >
      {firstItem}
      {additionalItems.length > 0 && (
        <OverflowPopover items={additionalItems} renderLabel={renderOverflowLabel} {...props} />
      )}
    </div>
  );
};

const getTagStyles = (theme: Theme): SerializedStyles => {
  const styles: Interpolation<Theme> = {
    marginRight: 0,
    color: theme.colors.actionTertiaryTextDefault,
    cursor: 'pointer',

    '&:focus': {
      color: theme.colors.actionTertiaryTextDefault,
    },

    '&:hover': {
      color: theme.colors.actionTertiaryTextHover,
    },

    '&:active': {
      color: theme.colors.actionTertiaryTextPress,
    },
  };

  return css(styles);
};
