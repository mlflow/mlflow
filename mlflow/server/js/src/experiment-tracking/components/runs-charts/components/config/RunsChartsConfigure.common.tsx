import { LegacySelect, Typography } from '@databricks/design-system';
import React, { ComponentProps, PropsWithChildren } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { makeCanonicalSortKey } from '../../../experiment-page/utils/experimentPage.common-utils';
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../common/utils/FeatureUtils';

/**
 * Represents a field in the compare run charts configuration modal.
 * Displays a title and content with proper margins.
 */
export const RunsChartsConfigureField = ({
  title,
  children,
}: PropsWithChildren<{
  title: string;
}>) => {
  return (
    <div
      css={(theme) => ({ marginBottom: theme.spacing.md * 2 })}
      data-testid="experiment-view-compare-runs-config-field"
    >
      <Typography.Title level={4}>{title}:</Typography.Title>
      {children}
    </div>
  );
};

/**
 * A searchable select for selecting metric or param from a categorized list.
 */
export const RunsChartsMetricParamSelect = ({
  value,
  onChange,
  metricKeyList,
  paramKeyList,
}: {
  value: string;
  onChange: ComponentProps<typeof LegacySelect>['onChange'];
  metricKeyList?: string[];
  paramKeyList?: string[];
}) => {
  const { formatMessage } = useIntl();

  const isEmpty = !paramKeyList?.length && !metricKeyList?.length;

  return (
    <LegacySelect
      css={styles.selectFull}
      value={
        isEmpty
          ? formatMessage({
              description:
                'Message displayed when no metrics or params are available in the compare runs chart configure modal',
              defaultMessage: 'No metrics or parameters available',
            })
          : value
      }
      disabled={isEmpty}
      onChange={onChange}
      dangerouslySetAntdProps={{ showSearch: true }}
    >
      {metricKeyList?.length ? (
        <LegacySelect.OptGroup
          label={formatMessage({
            defaultMessage: 'Metrics',
            description: "Label for 'metrics' option group in the compare runs chart configure modal",
          })}
        >
          {metricKeyList.map((metric) => (
            <LegacySelect.Option
              key={makeCanonicalSortKey('METRIC', metric)}
              value={makeCanonicalSortKey('METRIC', metric)}
            >
              {metric}
            </LegacySelect.Option>
          ))}
        </LegacySelect.OptGroup>
      ) : null}
      {paramKeyList?.length ? (
        <LegacySelect.OptGroup
          label={formatMessage({
            defaultMessage: 'Params',
            description: "Label for 'params' option group in the compare runs chart configure modal",
          })}
        >
          {paramKeyList.map((param) => (
            <LegacySelect.Option
              key={makeCanonicalSortKey('PARAM', param)}
              value={makeCanonicalSortKey('PARAM', param)}
            >
              {param}
            </LegacySelect.Option>
          ))}
        </LegacySelect.OptGroup>
      ) : null}
    </LegacySelect>
  );
};

export const RunsChartsRunNumberSelect = ({
  onChange,
  value,
  options,
}: {
  value?: number;
  onChange: ComponentProps<typeof LegacySelect<number>>['onChange'];
  options: (number | { value: number; label: React.ReactNode })[];
}) => {
  const { formatMessage } = useIntl();

  // After moving to the new run rows visibility model, we don't configure run count per chart
  if (shouldUseNewRunRowsVisibilityModel()) {
    return null;
  }
  return (
    <RunsChartsConfigureField
      title={formatMessage({
        defaultMessage: 'Max. no of runs shown',
        description: 'Label for the control allowing selection of number of runs displayed in a compare runs chart',
      })}
    >
      <LegacySelect<number> css={styles.selectFull} value={value} onChange={onChange}>
        {options.map((countOption) => {
          const optionValue = typeof countOption === 'number' ? countOption : countOption.value;
          const label = typeof countOption === 'number' ? countOption : countOption.label;
          return (
            <LegacySelect.Option key={optionValue} value={optionValue}>
              {label}
            </LegacySelect.Option>
          );
        })}
      </LegacySelect>
    </RunsChartsConfigureField>
  );
};

export const runsChartsRunCountDefaultOptions: { value: number; label: React.ReactNode }[] = [
  // We're not using any procedural generation so react-intl extractor can parse it
  {
    value: 5,
    label: (
      <FormattedMessage
        defaultMessage="5"
        description="Label for 5 first runs visible in run count selector within runs compare configuration modal"
      />
    ),
  },
  {
    value: 10,
    label: (
      <FormattedMessage
        defaultMessage="10"
        description="Label for 10 first runs visible in run count selector within runs compare configuration modal"
      />
    ),
  },
  {
    value: 20,
    label: (
      <FormattedMessage
        defaultMessage="20"
        description="Label for 20 first runs visible in run count selector within runs compare configuration modal"
      />
    ),
  },
];

const styles = { selectFull: { width: '100%' } };
