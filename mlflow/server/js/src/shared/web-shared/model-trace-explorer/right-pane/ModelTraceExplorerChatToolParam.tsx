import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceChatToolParamProperty } from '../ModelTrace.types';

export function ModelTraceExplorerChatToolParam({
  paramName,
  paramProperties,
  isRequired,
}: {
  paramName: string;
  paramProperties: ModelTraceChatToolParamProperty;
  isRequired: boolean;
}) {
  const { theme } = useDesignSystemTheme();

  const { type, description, enum: enumValues } = paramProperties;

  const hasAdditionalInfo = type || description || enumValues;

  const borderStyles = hasAdditionalInfo
    ? {
        borderTopLeftRadius: theme.borders.borderRadiusMd,
        borderTopRightRadius: theme.borders.borderRadiusMd,
        borderBottom: `1px solid ${theme.colors.border}`,
      }
    : {
        borderRadius: theme.borders.borderRadiusMd,
      };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          backgroundColor: theme.colors.backgroundSecondary,
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          gap: theme.spacing.sm,
          ...borderStyles,
        }}
      >
        <Typography.Title withoutMargins style={{ whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>
          {paramName}
        </Typography.Title>
        {isRequired && (
          <Typography.Text withoutMargins color="error">
            <FormattedMessage
              defaultMessage="required"
              description="Text displayed next to a function parameter to indicate that it is required"
            />
          </Typography.Text>
        )}
      </div>
      {hasAdditionalInfo && (
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: 'auto 1fr',
            gridTemplateRows: 'auto',
            gap: theme.spacing.md,
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          }}
        >
          {type && (
            <>
              <Typography.Text withoutMargins bold>
                <FormattedMessage
                  defaultMessage="Type"
                  description="Row heading in a table that contains the type of a function parameter (e.g. string, boolean)"
                />
              </Typography.Text>
              <Typography.Text withoutMargins code>
                {type}
              </Typography.Text>
            </>
          )}
          {description && (
            <>
              <Typography.Text withoutMargins bold>
                <FormattedMessage
                  defaultMessage="Description"
                  description="Row heading in a table that contains the description of a function parameter."
                />
              </Typography.Text>
              <Typography.Text withoutMargins>{description}</Typography.Text>
            </>
          )}
          {enumValues && (
            <>
              <Typography.Text withoutMargins bold>
                <FormattedMessage
                  defaultMessage="Enum Values"
                  description="Row heading in a table that contains the potential enum values that a function parameter can have."
                />
              </Typography.Text>
              <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
                {enumValues.map((value) => (
                  <Typography.Text withoutMargins code key={value}>
                    {value}
                  </Typography.Text>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
