import { useState } from 'react';

import * as Popover from './Popover';
import type { PopoverProps } from './Popover';
import { useDesignSystemTheme } from '../Hooks';
import type { IconProps } from '../Icon';
import { InfoIcon } from '../Icon';
import { useModalContext } from '../Modal';

export interface InfoPopoverProps extends React.HTMLAttributes<HTMLButtonElement> {
  popoverProps?: Omit<PopoverProps, 'children' | 'title'>;
  iconProps?: IconProps;
  iconTitle?: string;
  isKeyboardFocusable?: boolean;
  ariaLabel?: string;
}

export const InfoPopover = ({
  children,
  popoverProps,
  iconTitle,
  iconProps,
  isKeyboardFocusable = true,
  ariaLabel = 'More details',
}: InfoPopoverProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const { isInsideModal } = useModalContext();

  const [open, setOpen] = useState(false);

  const handleKeyDown = (event: React.KeyboardEvent<any>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      setOpen(!open);
    }
  };

  const { onKeyDown, ...restPopoverProps } = popoverProps || {};

  return (
    <Popover.Root
      componentId="codegen_design-system_src_design-system_popover_infopopover.tsx_36"
      open={open}
      onOpenChange={setOpen}
    >
      <Popover.Trigger asChild>
        <span
          style={{ display: 'inline-flex', cursor: 'pointer' }}
          // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
          tabIndex={isKeyboardFocusable ? 0 : -1}
          onKeyDown={handleKeyDown}
          aria-label={iconTitle ? undefined : ariaLabel}
          role="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setOpen(!open);
          }}
        >
          <InfoIcon
            aria-hidden={iconTitle ? false : true}
            title={iconTitle}
            aria-label={iconTitle}
            css={{
              color: theme.colors.textSecondary,
            }}
            {...iconProps}
          />
        </span>
      </Popover.Trigger>
      <Popover.Content
        align="start"
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            // If inside an AntD Modal, stop propagation of Escape key so that the modal doesn't close.
            // This is specifically for that case, so we only do it if inside a modal to limit the blast radius.
            if (isInsideModal) {
              e.stopPropagation();
              // If stopping propagation, we also need to manually close the popover since the radix
              // library expects the event to bubble up to the parent components.
              setOpen(false);
            }
          }
          onKeyDown?.(e);
        }}
        {...restPopoverProps}
      >
        {children}
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};
