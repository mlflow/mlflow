import React from 'react';

import type { TagColors } from '@databricks/design-system';
import {
  CheckCircleIcon,
  DangerIcon,
  Tag,
  Tooltip,
  useDesignSystemTheme,
  XCircleIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

// displays a single JSON-strigified assessment value as a tag
export const AssessmentDisplayValue = ({ jsonValue }: { jsonValue: string }) => {
  const { theme } = useDesignSystemTheme();

  // treat empty strings as null
  if (!jsonValue || jsonValue === '""') {
    return null;
  }

  let parsedValue: any;
  try {
    parsedValue = JSON.parse(jsonValue);
  } catch (e) {
    // if the value is not valid JSON, just use the string value
    parsedValue = jsonValue;
  }

  let color: TagColors = 'default';
  let children: React.ReactNode = JSON.stringify(parsedValue, null, 2);
  if (parsedValue === 'yes') {
    color = 'lime';
    children = (
      <>
        <CheckCircleIcon css={{ marginRight: theme.spacing.xs }} />
        <FormattedMessage defaultMessage="Yes" description="Label for an assessment with a 'yes' value" />
      </>
    );
  } else if (parsedValue === 'no') {
    color = 'coral';
    children = (
      <>
        <XCircleIcon css={{ marginRight: theme.spacing.xs }} />
        <FormattedMessage defaultMessage="No" description="Label for an assessment with a 'no' value" />
      </>
    );
  } else if (typeof parsedValue === 'string') {
    children = parsedValue;
  } else if (parsedValue === null) {
    // feedback can only have null values if they are errors
    color = 'coral';
    children = (
      <>
        <DangerIcon css={{ marginRight: theme.spacing.xs }} />
        <FormattedMessage defaultMessage="Error" description="Label for an assessment with an error" />
      </>
    );
  } else if (parsedValue === true) {
    color = 'lime';
    children = (
      <>
        <CheckCircleIcon css={{ marginRight: theme.spacing.xs }} />
        <FormattedMessage defaultMessage="True" description="Label for an assessment with a 'true' boolean value" />
      </>
    );
  } else if (parsedValue === false) {
    color = 'coral';
    children = (
      <>
        <XCircleIcon css={{ marginRight: theme.spacing.xs }} />
        <FormattedMessage defaultMessage="False" description="Label for an assessment with a 'false' boolean value" />
      </>
    );
  }

  return (
    <Tooltip componentId="shared.model-trace-explorer.assesment-value-tooltip" content={children}>
      <Tag
        css={{ display: 'inline-flex', maxWidth: '100%', minWidth: theme.spacing.md, marginRight: 0 }}
        componentId="shared.model-trace-explorer.assesment-value-tag"
        color={color}
      >
        <span
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            textWrap: 'nowrap',
          }}
        >
          {children}
        </span>
      </Tag>
    </Tooltip>
  );
};
