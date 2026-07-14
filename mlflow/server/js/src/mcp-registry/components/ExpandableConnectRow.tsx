import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Tag,
  Tooltip,
  Typography,
  VisibleIcon,
  VisibleOffIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  expandableRowButtonStyles,
  chevronContainerStyles,
  expandedContentPanelStyles,
  noShrinkStyles,
} from '../styles';

export const ExpandableConnectRow = ({
  expanded,
  onToggle,
  showTopBorder,
  showVisibilityControls,
  isHidden,
  onToggleVisibility,
  ariaLabel,
  visibilityAriaLabel,
  children,
  expandedContent,
}: {
  expanded: boolean;
  onToggle: () => void;
  showTopBorder: boolean;
  showVisibilityControls?: boolean;
  isHidden?: boolean;
  onToggleVisibility?: (visible: boolean) => void;
  ariaLabel: string;
  visibilityAriaLabel: string;
  children: React.ReactNode;
  expandedContent: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const isVisible = !isHidden;
  const isDisabled = !isVisible;

  if (isDisabled && !showVisibilityControls) return null;

  return (
    <div
      css={{
        borderTop: showTopBorder ? `1px solid ${theme.colors.border}` : 'none',
        opacity: isDisabled ? 0.5 : 1,
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-label={ariaLabel}
        css={expandableRowButtonStyles(theme)}
      >
        <div css={chevronContainerStyles(theme)}>{expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}</div>
        {children}
        {showVisibilityControls && isDisabled && (
          <Tag
            componentId="mlflow.mcp_registry.detail.connect_option.disabled_tag"
            color="charcoal"
            css={noShrinkStyles}
          >
            <FormattedMessage defaultMessage="Disabled" description="Label for disabled connect option" />
          </Tag>
        )}
        {showVisibilityControls && (
          <div css={noShrinkStyles} onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
            <Tooltip
              componentId="mlflow.mcp_registry.detail.connect_option.visibility_tooltip"
              content={
                isVisible ? (
                  <FormattedMessage
                    defaultMessage="Visible to developers. Click to hide."
                    description="Tooltip for visible connect option toggle"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Hidden from developers. Click to show."
                    description="Tooltip for hidden connect option toggle"
                  />
                )
              }
            >
              <Button
                componentId="mlflow.mcp_registry.detail.connect_option.visibility_toggle"
                type="tertiary"
                size="small"
                icon={isVisible ? <VisibleIcon /> : <VisibleOffIcon />}
                onClick={() => onToggleVisibility?.(!!isHidden)}
                aria-label={visibilityAriaLabel}
              />
            </Tooltip>
          </div>
        )}
      </button>

      {expanded && <div css={expandedContentPanelStyles(theme)}>{expandedContent}</div>}
    </div>
  );
};
