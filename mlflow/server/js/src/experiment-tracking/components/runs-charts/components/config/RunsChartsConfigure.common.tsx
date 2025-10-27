import {
  LegacySelect,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ComponentProps, PropsWithChildren } from 'react';
import React, { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { makeCanonicalSortKey } from '../../../experiment-page/utils/experimentPage.common-utils';

/**
 * Represents a field in the compare run charts configuration modal.
 * Displays a title and content with proper margins.
 */
export const RunsChartsConfigureField = ({
  title,
  compact = false,
  children,
}: PropsWithChildren<{
  title: React.ReactNode;
  compact?: boolean;
}>) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{ marginBottom: compact ? theme.spacing.sm : theme.spacing.md * 2 }}
      data-testid="experiment-view-compare-runs-config-field"
    >
      <Typography.Title level={4}>{title}</Typography.Title>
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

export const RunsChartsMetricParamSelectV2 = ({
  value,
  id,
  onChange,
  metricOptions = [],
  paramOptions = [],
}: {
  value: string;
  id: string;
  onChange: (value: string) => void;
  metricOptions: {
    key: string;
    datasetName: string | undefined;
    metricKey: string;
  }[];
  paramOptions: {
    key: string;
    paramKey: string;
  }[];
}) => {
  const { formatMessage } = useIntl();

  const isEmpty = !paramOptions.length && !metricOptions.length;

  return (
    <SimpleSelect
      componentId="mlflow.charts.chart_configure.metric_with_dataset_select"
      id={id}
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
      onChange={({ target }) => {
        onChange(target.value);
      }}
      contentProps={{
        matchTriggerWidth: true,
        maxHeight: 500,
      }}
    >
      {metricOptions?.length ? (
        <SimpleSelectOptionGroup
          label={formatMessage({
            defaultMessage: 'Metrics',
            description: "Label for 'metrics' option group in the compare runs chart configure modal",
          })}
        >
          {metricOptions.map(({ datasetName, key, metricKey }) => (
            <SimpleSelectOption key={key} value={key}>
              {datasetName && (
                <Tag componentId="mlflow.charts.chart_configure.metric_with_dataset_select.tag">{datasetName}</Tag>
              )}{' '}
              {metricKey}
            </SimpleSelectOption>
          ))}
        </SimpleSelectOptionGroup>
      ) : null}
      {paramOptions?.length ? (
        <SimpleSelectOptionGroup
          label={formatMessage({
            defaultMessage: 'Params',
            description: "Label for 'params' option group in the compare runs chart configure modal",
          })}
        >
          {paramOptions.map(({ key, paramKey }) => (
            <SimpleSelectOption key={key} value={key}>
              {paramKey}
            </SimpleSelectOption>
          ))}
        </SimpleSelectOptionGroup>
      ) : null}
    </SimpleSelect>
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
