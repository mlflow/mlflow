import { Button, Tag, Tooltip, VisibleIcon, VisibleOffIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { noShrinkStyles } from '../styles';

export const VisibilityToggle = ({
  componentId,
  isVisible,
  onToggle,
  ariaLabel,
  showDisabledTag,
}: {
  componentId: string;
  isVisible: boolean;
  onToggle: (nowVisible: boolean) => void;
  ariaLabel: string;
  showDisabledTag?: boolean;
}) => (
  <>
    {showDisabledTag && !isVisible && (
      <Tag componentId={`${componentId}.disabled_tag`} color="charcoal" css={noShrinkStyles}>
        <FormattedMessage defaultMessage="Disabled" description="Label for disabled connect option" />
      </Tag>
    )}
    <div css={noShrinkStyles} onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
      <Tooltip
        componentId={`${componentId}.visibility_tooltip`}
        content={
          isVisible ? (
            <FormattedMessage
              defaultMessage="Shown in Connect tab. Click to hide."
              description="Tooltip for visible connect option toggle"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Hidden from Connect tab. Click to show."
              description="Tooltip for hidden connect option toggle"
            />
          )
        }
      >
        <Button
          componentId={`${componentId}.visibility_row`}
          type="tertiary"
          size="small"
          icon={isVisible ? <VisibleIcon /> : <VisibleOffIcon />}
          onClick={() => onToggle(!isVisible)}
          aria-label={ariaLabel}
        />
      </Tooltip>
    </div>
  </>
);
