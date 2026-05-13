import { Typography } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import React from 'react';

export interface DescriptionsProps {
  columns?: number;
}

export interface DescriptionsItemProps {
  label: string | React.ReactNode;
  labelSize?: 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
  span?: number;
}

/**
 * A component that displays the informative data in a key-value
 * fashion. Behaves similarly to antd's <Descriptions /> component.
 * If the number of columns is specified, then the key-values will
 * be displayed as such and will always be that number of columns
 * regardless of the width of the window.
 * If the number of columns is not specified, then the number of
 * columns will vary based on the size of the window.
 *
 * The following example will display four key-value descriptions
 * using two columns, which will result in data displayed in two rows:
 *
 * @example
 * <Descriptions columns={2}>
 *   <Descriptions.Item label="The label">The value</Descriptions.Item>
 *   <Descriptions.Item label="Another label">Another value</Descriptions.Item>
 *   <Descriptions.Item label="A label">A value</Descriptions.Item>
 *   <Descriptions.Item label="Extra label">Extra value</Descriptions.Item>
 * </Descriptions>
 */
export const Descriptions = ({ children, columns }: React.PropsWithChildren<DescriptionsProps>) => {
  const instanceStyles = columns ? styles.descriptionsArea(columns) : styles.autoFitArea;

  return <div css={instanceStyles}>{children}</div>;
};

Descriptions.Item = ({ label, labelSize = 'sm', children, span }: React.PropsWithChildren<DescriptionsItemProps>) => {
  return (
    <div data-testid="descriptions-item" css={styles.descriptionItem(span || 1)}>
      <div data-testid="descriptions-item-label" css={{ whiteSpace: 'nowrap' }}>
        <Typography.Text size={labelSize} color="secondary">
          {label}
        </Typography.Text>
      </div>
      <div data-testid="descriptions-item-colon" css={styles.colon}>
        <Typography.Text size={labelSize} color="secondary">
          :
        </Typography.Text>
      </div>
      <div data-testid="descriptions-item-content">{children}</div>
    </div>
  );
};

const styles = {
  descriptionsArea: (columnCount: number) => (theme: Theme) => ({
    display: 'grid',
    gridTemplateColumns: `repeat(${columnCount}, minmax(100px, 1fr))`,
    columnGap: theme.spacing.sm,
    rowGap: theme.spacing.md,
    marginBottom: theme.spacing.lg,
  }),
  autoFitArea: (theme: Theme) => ({
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
    gridGap: theme.spacing.md,
  }),
  descriptionItem: (span: number) => ({
    display: 'flex',
    gridColumn: `span ${span}`,
  }),
  colon: {
    margin: '0 8px 0 0',
  },
};
