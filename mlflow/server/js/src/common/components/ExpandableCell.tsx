import { useMemo } from 'react';
import { Button, ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';

export const ExpandedJSONValueCell = ({ value }: { value: string }) => {
  const structuredJSONValue = useMemo(() => {
    // Attempts to parse the value as JSON and returns a pretty printed version if successful.
    // If JSON structure is not found, returns null.
    try {
      const objectData = JSON.parse(value);
      return JSON.stringify(objectData, null, 2);
    } catch (e) {
      return null;
    }
  }, [value]);
  return (
    <div
      css={{
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontFamily: structuredJSONValue ? 'monospace' : undefined,
      }}
    >
      {structuredJSONValue || value}
    </div>
  );
};

const ExpandableCell = ({
  value,
  isExpanded,
  toggleExpanded,
  hideCollapseButton,
}: {
  value: string;
  isExpanded: boolean;
  toggleExpanded: () => void;
  hideCollapseButton?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.xs,
      }}
    >
      {!hideCollapseButton && (
        <Button
          componentId="mlflow.common.expandable_cell"
          size="small"
          icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => toggleExpanded()}
          css={{ flexShrink: 0 }}
        />
      )}
      <div
        title={value}
        css={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitBoxOrient: 'vertical',
          WebkitLineClamp: isExpanded ? undefined : '3',
        }}
      >
        {isExpanded ? <ExpandedJSONValueCell value={value} /> : value}
      </div>
    </div>
  );
};
