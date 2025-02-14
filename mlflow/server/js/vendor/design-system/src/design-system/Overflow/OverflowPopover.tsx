import { useState } from 'react';

import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks';
import { LegacyTooltip } from '../LegacyTooltip';
import { Popover } from '../Popover';
import type { PopoverProps } from '../Popover/Popover';
import { addDebugOutlineIfEnabled } from '../utils/debug';

interface OverflowPopoverProps extends PopoverProps {
  items: React.ReactNode[];
  renderLabel?: (label: string) => React.ReactNode;
  tooltipText?: string;
}

export const OverflowPopover = ({ items, renderLabel, tooltipText, ...props }: OverflowPopoverProps) => {
  const { theme } = useDesignSystemTheme();
  const [showTooltip, setShowTooltip] = useState(true);
  const label = `+${items.length}`;

  let trigger = (
    <span css={{ lineHeight: 0 }} {...addDebugOutlineIfEnabled()}>
      <Popover.Trigger asChild>
        {/* button is needed so that the popover will open when the trigger is clicked */}
        <Button componentId="something" type="link">
          {renderLabel ? renderLabel(label) : label}
        </Button>
      </Popover.Trigger>
    </span>
  );

  if (showTooltip) {
    trigger = <LegacyTooltip title={tooltipText || 'See more items'}>{trigger}</LegacyTooltip>;
  }

  return (
    <Popover.Root
      componentId="codegen_design-system_src_design-system_overflow_overflowpopover.tsx_37"
      onOpenChange={(open) => setShowTooltip(!open)}
    >
      {trigger}
      <Popover.Content align="start" {...props} {...addDebugOutlineIfEnabled()}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {items.map((item, index) => (
            <div key={`overflow-${index}`}>{item}</div>
          ))}
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
