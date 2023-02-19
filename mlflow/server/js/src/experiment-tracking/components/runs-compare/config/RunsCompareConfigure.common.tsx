import { Select, Typography } from '@databricks/design-system';
import React, { ComponentProps, PropsWithChildren } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { makeCanonicalSortKey } from '../../experiment-page/utils/experimentPage.column-utils';

/**
 * Represents a field in the compare run charts configuration modal.
 * Displays a title and content with proper margins.
 */
export const RunsCompareConfigureField = ({
  title,
  children,
}: PropsWithChildren<{
  title: string;
}>) => {
  return (
    <div
      css={(theme) => ({ marginBottom: theme.spacing.md * 2 })}
      data-testid='experiment-view-compare-runs-config-field'
    >
      <Typography.Title level={4}>{title}:</Typography.Title>
      {children}
    </div>
  );
};

/**
 * A searchable select for selecting metric or param from a categorized list.
 */
export const RunsCompareMetricParamSelect = ({
  value,
  onChange,
  metricKeyList,
  paramKeyList,
}: {
  value: string;
  onChange: ComponentProps<typeof Select>['onChange'];
  metricKeyList?: string[];
  paramKeyList?: string[];
}) => {
  const { formatMessage } = useIntl();

  const isEmpty = !paramKeyList?.length && !metricKeyList?.length;

  return (
    <Select
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
        <Select.OptGroup
          label={formatMessage({
            defaultMessage: 'Metrics',
            description:
              "Label for 'metrics' option group in the compare runs chart configure modal",
          })}
        >
          {metricKeyList.map((metric) => (
            <Select.Option
              key={makeCanonicalSortKey('METRIC', metric)}
              value={makeCanonicalSortKey('METRIC', metric)}
            >
              {metric}
            </Select.Option>
          ))}
        </Select.OptGroup>
      ) : null}
      {paramKeyList?.length ? (
        <Select.OptGroup
          label={formatMessage({
            defaultMessage: 'Params',
            description:
              "Label for 'params' option group in the compare runs chart configure modal",
          })}
        >
          {paramKeyList.map((param) => (
            <Select.Option
              key={makeCanonicalSortKey('PARAM', param)}
              value={makeCanonicalSortKey('PARAM', param)}
            >
              {param}
            </Select.Option>
          ))}
        </Select.OptGroup>
      ) : null}
    </Select>
  );
};

export const RunsCompareRunNumberSelect = ({
  onChange,
  value,
  options,
}: {
  value?: number;
  onChange: ComponentProps<typeof Select<number>>['onChange'];
  options: (number | { value: number; label: React.ReactNode })[];
}) => {
  const { formatMessage } = useIntl();
  return (
    <RunsCompareConfigureField
      title={formatMessage({
        defaultMessage: 'Max. no of runs shown',
        description:
          'Label for the control allowing selection of number of runs displayed in a compare runs chart',
      })}
    >
      <Select<number> css={styles.selectFull} value={value} onChange={onChange}>
        {options.map((countOption) => {
          const optionValue = typeof countOption === 'number' ? countOption : countOption.value;
          const label = typeof countOption === 'number' ? countOption : countOption.label;
          return (
            <Select.Option key={optionValue} value={optionValue}>
              {label}
            </Select.Option>
          );
        })}
      </Select>
    </RunsCompareConfigureField>
  );
};

export const runsCompareRunCountDefaultOptions: { value: number; label: React.ReactNode }[] = [
  // We're not using any procedural generation so react-intl extractor can parse it
  {
    value: 5,
    label: (
      <FormattedMessage
        defaultMessage='5'
        description='Label for 5 first runs visible in run count selector within runs compare configuration modal'
      />
    ),
  },
  {
    value: 10,
    label: (
      <FormattedMessage
        defaultMessage='10'
        description='Label for 10 first runs visible in run count selector within runs compare configuration modal'
      />
    ),
  },
  {
    value: 20,
    label: (
      <FormattedMessage
        defaultMessage='20'
        description='Label for 20 first runs visible in run count selector within runs compare configuration modal'
      />
    ),
  },
];

const styles = { selectFull: { width: '100%' } };
