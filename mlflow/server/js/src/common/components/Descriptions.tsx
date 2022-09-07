import { Typography } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useMemo } from 'react';

export interface DescriptionsProps {
  columns?: number;
}

export interface DescriptionsItemProps {
  label: string | React.ReactNode;
}

/**
 * A component that displays the informative data in a key-value
 * fashion. Behaves similarly to antd's <Descriptions /> component.
 *
 * The following example will display four key-value descriptions
 * using two colums, which will result in data displayed in two rows:
 *
 * @example
 * <Descriptions columns={2}>
 *   <Descriptions.Item label="The label">The value</Descriptions.Item>
 *   <Descriptions.Item label="Another label">Another value</Descriptions.Item>
 *   <Descriptions.Item label="A label">A value</Descriptions.Item>
 *   <Descriptions.Item label="Extra label">Extra value</Descriptions.Item>
 * </Descriptions>
 */
export const Descriptions = ({
  children,
  columns = 3,
}: React.PropsWithChildren<DescriptionsProps>) => {
  const instanceStyles = useMemo(() => styles.descriptionsArea(columns), [columns]);

  return <div css={instanceStyles}>{children}</div>;
};

Descriptions.Item = ({ label, children }: React.PropsWithChildren<DescriptionsItemProps>) => {
  return (
    <>
      <div>
        <Typography.Text size='sm' color='secondary'>
          {label}:
        </Typography.Text>
      </div>
      <div>{children}</div>
    </>
  );
};

const styles = {
  descriptionsArea: (columnCount: number) => (theme: Theme) => ({
    display: 'grid',
    gridTemplateColumns: `repeat(${columnCount}, auto 1fr)`,
    columnGap: theme.spacing.sm,
    rowGap: theme.spacing.md,
    marginBottom: theme.spacing.lg,
  }),
};
