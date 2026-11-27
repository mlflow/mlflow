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

const BUILTIN_SCORER_ASSESSMENT_DISPLAY = {
  user_frustration: {
    none: {
      color: 'lime' as TagColors,
      icon: CheckCircleIcon,
      label: (
        <FormattedMessage
          defaultMessage="None"
          description="Label for a user_frustration assessment with 'none' value"
        />
      ),
    },
    resolved: {
      color: 'lemon' as TagColors,
      icon: CheckCircleIcon,
      label: (
        <FormattedMessage
          defaultMessage="Resolved"
          description="Label for a user_frustration assessment with 'resolved' value"
        />
      ),
    },
    unresolved: {
      color: 'coral' as TagColors,
      icon: XCircleIcon,
      label: (
        <FormattedMessage
          defaultMessage="Unresolved"
          description="Label for a user_frustration assessment with 'unresolved' value"
        />
      ),
    },
  },
} as const;

type BuiltInScorerAssessmentName = keyof typeof BUILTIN_SCORER_ASSESSMENT_DISPLAY;

const getBuiltInScorerAssessmentDisplay = (
  assessmentName: string | undefined,
  parsedValue: any,
  theme: any,
  skipIcons: boolean,
): { color: TagColors; children: React.ReactNode } | undefined => {
  if (!assessmentName) {
    return undefined;
  }

  const builtInAssessment = BUILTIN_SCORER_ASSESSMENT_DISPLAY[assessmentName as BuiltInScorerAssessmentName];
  if (!builtInAssessment) {
    return undefined;
  }

  const valueKey = String(parsedValue ?? '') as keyof typeof builtInAssessment;
  const displayConfig = builtInAssessment[valueKey];
  if (!displayConfig) {
    return undefined;
  }

  const IconComponent = displayConfig.icon;
  return {
    color: displayConfig.color,
    children: (
      <>
        {!skipIcons && <IconComponent css={{ marginRight: theme.spacing.xs }} />}
        {displayConfig.label}
      </>
    ),
  };
};

// displays a single JSON-strigified assessment value as a tag
export const AssessmentDisplayValue = ({
  jsonValue,
  className,
  prefix,
  skipIcons = false,
  overrideColor,
  assessmentName,
  disableTooltip = false,
}: {
  jsonValue: string;
  className?: string;
  prefix?: React.ReactNode;
  skipIcons?: boolean;
  overrideColor?: TagColors;
  assessmentName?: string;
  // Skip the value tooltip (e.g. when a caller already reveals the full value in a hover card).
  disableTooltip?: boolean;
}) => {
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

  const builtInDisplay = getBuiltInScorerAssessmentDisplay(assessmentName, parsedValue, theme, skipIcons);
  if (builtInDisplay) {
    color = builtInDisplay.color;
    children = builtInDisplay.children;
  } else if (parsedValue === 'yes') {
    color = 'lime';
    children = (
      <>
        {!skipIcons && <CheckCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="Yes" description="Label for an assessment with a 'yes' value" />
      </>
    );
  } else if (parsedValue === 'no') {
    color = 'coral';
    children = (
      <>
        {!skipIcons && <XCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="No" description="Label for an assessment with a 'no' value" />
      </>
    );
  } else if (parsedValue === 'pass') {
    color = 'lime';
    children = (
      <>
        {!skipIcons && <CheckCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="Pass" description="Label for an assessment with a 'pass' value" />
      </>
    );
  } else if (parsedValue === 'fail') {
    color = 'coral';
    children = (
      <>
        {!skipIcons && <XCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="Fail" description="Label for an assessment with a 'fail' value" />
      </>
    );
  } else if (parsedValue === 'good') {
    color = 'lime';
    children = (
      <>
        {!skipIcons && <CheckCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="Good" description="Label for an assessment with a 'good' value" />
      </>
    );
  } else if (parsedValue === 'bad') {
    color = 'coral';
    children = (
      <>
        {!skipIcons && <XCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="Bad" description="Label for an assessment with a 'bad' value" />
      </>
    );
  } else if (parsedValue === true || parsedValue === 'true') {
    // Match both the JSON boolean and its string form so a stringified `true` gets the same treatment.
    color = 'lime';
    children = (
      <>
        {!skipIcons && <CheckCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="True" description="Label for an assessment with a 'true' boolean value" />
      </>
    );
  } else if (parsedValue === false || parsedValue === 'false') {
    color = 'coral';
    children = (
      <>
        {!skipIcons && <XCircleIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="False" description="Label for an assessment with a 'false' boolean value" />
      </>
    );
  } else if (typeof parsedValue === 'string') {
    children = parsedValue;
  } else if (parsedValue === null) {
    // feedback can only have null values if they are errors
    color = 'coral';
    children = (
      <>
        {!skipIcons && <DangerIcon css={{ marginRight: theme.spacing.xs }} />}
        <FormattedMessage defaultMessage="Error" description="Label for an assessment with an error" />
      </>
    );
  }

  const tooltipContent = (
    <>
      {prefix}
      {children}
    </>
  );

  const tag = (
    <Tag
      css={{ display: 'inline-flex', maxWidth: '100%', minWidth: theme.spacing.md, marginRight: 0 }}
      componentId="shared.model-trace-explorer.assesment-value-tag"
      color={overrideColor ?? color}
      className={className}
    >
      <span
        css={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          textWrap: 'nowrap',
        }}
      >
        {prefix}
        {children}
      </span>
    </Tag>
  );

  if (disableTooltip) {
    return tag;
  }

  return (
    <Tooltip componentId="shared.model-trace-explorer.assesment-value-tooltip" content={tooltipContent}>
      {tag}
    </Tooltip>
  );
};
