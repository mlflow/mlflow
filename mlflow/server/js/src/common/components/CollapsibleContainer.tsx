import type { Dispatch, ReactNode, SetStateAction } from 'react';
import { useLayoutEffect, useRef, useState } from 'react';

import { Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { toRGBA } from '../utils/toRGBA';

interface CollapsibleContainerProps {
  children: ReactNode;
  maxHeight?: number;
  isExpanded: boolean;
  setIsExpanded: Dispatch<SetStateAction<boolean>>;
}

export const CollapsibleContainer = ({
  children,
  maxHeight = 150,
  setIsExpanded,
  isExpanded,
}: CollapsibleContainerProps) => {
  const { theme } = useDesignSystemTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const [isCollapsible, setIsCollapsible] = useState(false);
  const intl = useIntl();

  useLayoutEffect(() => {
    if (!containerRef.current) return;

    const checkCollapsible = () => {
      if (containerRef.current) {
        const currentHeight = containerRef.current.scrollHeight;
        const shouldBeCollapsible = currentHeight > maxHeight;
        setIsCollapsible(shouldBeCollapsible);
      }
    };

    const resizeObserver = new ResizeObserver(() => {
      checkCollapsible();
    });

    resizeObserver.observe(containerRef.current);
    checkCollapsible();

    return () => {
      resizeObserver.disconnect();
    };
  }, [maxHeight, setIsExpanded]);

  const containerStyles = {
    maxHeight: isExpanded ? 'none' : `${maxHeight}px`,
    overflow: isExpanded ? 'visible' : 'hidden',
    position: 'relative' as const,
    width: '100%',
  };

  return (
    <div>
      <div css={containerStyles} ref={containerRef}>
        {children}
        {!isExpanded && isCollapsible && (
          <div
            data-testid="truncation-gradient"
            css={{
              position: 'absolute',
              bottom: 0,
              height: '60%',
              width: '100%',
              background: `linear-gradient(to bottom, ${toRGBA(theme.colors.backgroundPrimary, 0)}, ${toRGBA(
                theme.colors.backgroundPrimary,
                1,
              )})`,
            }}
          />
        )}
      </div>
      {isCollapsible && (
        <Button
          type="link"
          componentId="discovery.data_explorer.entity_comment.show_comment_text_toggle"
          data-testid="show-comment-toggle"
          onClick={() => setIsExpanded((prev) => !prev)}
          aria-label={
            isExpanded
              ? intl.formatMessage({
                  defaultMessage: 'Collapse description',
                  description: 'Aria label for button that collapses a long description',
                })
              : intl.formatMessage({
                  defaultMessage: 'Expand description',
                  description: 'Aria label for button that expands a collapsed long description',
                })
          }
          style={{ marginTop: theme.spacing.xs }}
        >
          {isExpanded ? (
            <FormattedMessage
              defaultMessage="Show less"
              description="Button text to show less description text for the entity"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Show more"
              description="Button text to show more description text for the entity"
            />
          )}
        </Button>
      )}
    </div>
  );
};
