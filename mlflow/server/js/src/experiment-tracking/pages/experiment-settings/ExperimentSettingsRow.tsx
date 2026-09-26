import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ReactNode } from 'react';
import { useMemo } from 'react';
import { v4 as uuidv4 } from 'uuid';

const SETTING_CELL_CONTAINER_NAME = 'SettingCell';
const STACKED_BREAKPOINT = 480;
const STACKED_QUERY = `@container ${SETTING_CELL_CONTAINER_NAME} (width < ${STACKED_BREAKPOINT}px)`;

export interface ExperimentSettingsRowProps {
  children: ReactNode | ((labelId: string) => ReactNode);
  description?: ReactNode;
  label: ReactNode;
}

export const ExperimentSettingsRow = ({ children, description, label }: ExperimentSettingsRowProps) => {
  const { theme } = useDesignSystemTheme();
  const labelId = useMemo(() => uuidv4(), []);

  return (
    <div
      css={{
        containerName: SETTING_CELL_CONTAINER_NAME,
        containerType: 'inline-size',
      }}
    >
      <section
        className="__SettingCell__"
        css={{
          alignItems: 'center',
          borderBlockEnd: `1px solid ${theme.colors.border}`,
          columnGap: theme.spacing.lg,
          display: 'grid',
          gridTemplate: '"label value" / 1fr fit-content(50%)',
          margin: 0,
          paddingBlock: theme.spacing.lg,
          [STACKED_QUERY]: {
            gridTemplate: '"label" "value" / 100%',
          },
        }}
      >
        <div css={{ gridArea: 'label' }}>
          <Typography.Text id={labelId} bold>
            {label}
          </Typography.Text>
          {description && <Typography.Hint>{description}</Typography.Hint>}
        </div>
        <div
          className="__ExperimentSettingsValue__"
          css={{
            alignItems: 'center',
            display: 'flex',
            gap: theme.spacing.xs,
            gridArea: 'value',
            justifyContent: 'flex-end',
            minWidth: 0,
            overflowWrap: 'anywhere',
            textAlign: 'right',
            width: '100%',
            '& > *': { maxWidth: '100%', minWidth: 0 },
            [STACKED_QUERY]: {
              marginBlockStart: theme.spacing.sm,
              paddingBlockStart: theme.spacing.sm,
            },
          }}
        >
          {typeof children === 'function' ? children(labelId) : children}
        </div>
      </section>
    </div>
  );
};
