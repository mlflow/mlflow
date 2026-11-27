import { useDesignSystemTheme } from '@databricks/design-system';
import type { ReactNode } from 'react';
import {
  SETTING_CELL_CONTAINER_NAME,
  SETTING_CELL_STACKED_BREAKPOINT,
  containerQueryWithFallback,
} from './SettingCell';

const VALUE_CLASS_NAME = '__ExperimentSettingsValue__';

export interface ExperimentSettingsValueProps {
  children: ReactNode;
}

/** Right-aligned value slot of a settings row; left-aligned below the label when the row is stacked. */
export const ExperimentSettingsValue = ({ children }: ExperimentSettingsValueProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      className={VALUE_CLASS_NAME}
      css={{
        alignItems: 'center',
        display: 'flex',
        gap: theme.spacing.xs,
        justifyContent: 'flex-end',
        minWidth: 0,
        overflowWrap: 'anywhere',
        textAlign: 'right',
        width: '100%',
        '& > *': {
          maxWidth: '100%',
          minWidth: 0,
        },
        ...containerQueryWithFallback(
          SETTING_CELL_CONTAINER_NAME,
          VALUE_CLASS_NAME,
          `(width < ${SETTING_CELL_STACKED_BREAKPOINT}px)`,
          { paddingBlockStart: theme.spacing.sm },
        ),
      }}
    >
      {children}
    </div>
  );
};
