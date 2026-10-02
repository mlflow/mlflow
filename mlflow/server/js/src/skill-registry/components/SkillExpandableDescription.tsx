import { useLayoutEffect, useRef, useState } from 'react';
import { Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const SkillExpandableDescription = ({ text }: { text: string }) => {
  const { theme } = useDesignSystemTheme();
  const textRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);

  useLayoutEffect(() => {
    setExpanded(false);
  }, [text]);

  useLayoutEffect(() => {
    const element = textRef.current;
    if (!element || expanded) return;
    setOverflows(element.scrollHeight > element.clientHeight + 1);
  }, [text, expanded]);

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
