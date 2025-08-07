import { isString } from 'lodash';
import React, { useMemo } from 'react';

import { CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { EvaluationsReviewCopyButton } from './EvaluationsReviewCopyButton';
import { EvaluationsReviewExpandedJSONValueCell } from './EvaluationsReviewExpandableCell';
import { EXPECTED_FACTS_FIELD_NAME, stringifyValue } from './GenAiEvaluationTracesReview.utils';
import { useMarkdownConverter } from '../utils/MarkdownUtils';

export const EvaluationsReviewTextBox = ({
  fieldName,
  title,
  value,
  showCopyIcon,
}: {
  fieldName: string;
  title: React.ReactNode;
  value: any;
  showCopyIcon?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  const { makeHTML } = useMarkdownConverter();

  const htmlContent = useMemo(() => {
    return isString(value) ? makeHTML(value) : null;
  }, [value, makeHTML]);

  const jsonContent = useMemo(() => {
    return isString(value) ? null : stringifyValue(value);
  }, [value]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        flex: 1,
        border: `1px solid ${theme.colors.border}`,
        padding: theme.spacing.md,
        borderRadius: theme.general.borderRadiusBase,
        marginBottom: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Text bold>{title}</Typography.Text>
        {showCopyIcon && (
          <EvaluationsReviewCopyButton
            copyText={stringifyValue(value)}
            showLabel={false}
            type="tertiary"
            icon={<CopyIcon />}
          />
        )}
      </div>
      <Typography.Paragraph
        css={{
          marginBottom: '0 !important',
        }}
      >
        {isString(value) ? (
          // eslint-disable-next-line react/no-danger
          <span css={{ display: 'contents' }} dangerouslySetInnerHTML={{ __html: htmlContent ?? '' }} />
        ) : fieldName === EXPECTED_FACTS_FIELD_NAME && Array.isArray(value) ? (
          <ul>
            {value.map((fact, index) => (
              <li key={index}>
                <EvaluationsReviewExpandedJSONValueCell key={index} value={fact} />
              </li>
            ))}
          </ul>
        ) : (
          <EvaluationsReviewExpandedJSONValueCell value={jsonContent} />
        )}
      </Typography.Paragraph>
    </div>
  );
};
