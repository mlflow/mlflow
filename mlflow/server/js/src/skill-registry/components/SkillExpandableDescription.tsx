import { useLayoutEffect, useRef, useState } from 'react';
import { Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useResizeObserver } from '@databricks/web-shared/hooks';

/** Callers key this by `text` so a new description starts collapsed. */
export const SkillExpandableDescription = ({ text }: { text: string }) => {
  const { theme } = useDesignSystemTheme();
  const textRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  // Resizing the page or sidebar can make a short description wrap past the clamp.
  const width = useResizeObserver({ ref: textRef })?.width;

  useLayoutEffect(() => {
    const element = textRef.current;
    if (!element || expanded) return;
    setOverflows(element.scrollHeight > element.clientHeight + 1);
  }, [text, expanded, width]);

  return (
    <div css={{ marginTop: theme.spacing.xs }}>
      <div
        ref={textRef}
        css={{
          color: theme.colors.textSecondary,
          lineHeight: 1.5,
          overflow: 'hidden',
          maxHeight: expanded ? 'none' : '3em',
        }}
      >
        {text}
      </div>
      {(overflows || expanded) && (
        <Button
          componentId="mlflow.skill_registry.detail.description_toggle"
          type="link"
          size="small"
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? (
            <FormattedMessage defaultMessage="Show less" description="Collapses a long skill description" />
          ) : (
            <FormattedMessage defaultMessage="Read more" description="Expands a long skill description" />
          )}
        </Button>
      )}
    </div>
  );
};
