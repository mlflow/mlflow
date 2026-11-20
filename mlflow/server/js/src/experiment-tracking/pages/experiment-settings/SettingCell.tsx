import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { CSSObject } from '@emotion/react';
import { useMemo } from 'react';
import type { ReactNode } from 'react';
import { v4 as uuidv4 } from 'uuid';

export const SETTING_CELL_CONTAINER_NAME = 'SettingCell';
export const SETTING_CELL_STACKED_BREAKPOINT = 480;
export const SETTING_CELL_CLASS_NAME = '__SettingCell__';
const SETTING_CELL_VALUE_CLASS_NAME = `${SETTING_CELL_CLASS_NAME}-value`;

/** A stable, document-unique element ID for ARIA relationships. */
export const useElementId = (prefix: string) => useMemo(() => `${prefix}-${uuidv4()}`, [prefix]);

/**
 * Applies `css` when the named container matches `condition`, falling back to a media query in
 * browsers without container queries.
 */
export const containerQueryWithFallback = (
  containerName: string,
  className: string,
  condition: string,
  css: CSSObject,
): CSSObject => ({
  [`@container ${containerName} ${condition}`]: {
    [`&.${className}`]: css,
  },
  [`@supports not (container-type: size)`]: {
    [`@media ${condition}`]: css,
  },
});

interface SettingCellProps {
  as?: 'div' | 'section';
  label: ReactNode;
  value: ReactNode;
}

/** One settings row: label (with an optional hint) on the left, value on the right; stacked when narrow. */
const SettingCellRoot = ({ as: Component = 'div', label, value }: SettingCellProps) => {
  const { theme } = useDesignSystemTheme();
  const stacked = `(width < ${SETTING_CELL_STACKED_BREAKPOINT}px)`;

  return (
    <div css={{ containerName: SETTING_CELL_CONTAINER_NAME, containerType: 'inline-size' }}>
      <Component
        className={SETTING_CELL_CLASS_NAME}
        css={{
          alignItems: 'center',
          borderBlockEnd: `1px solid ${theme.colors.border}`,
          columnGap: theme.spacing.lg,
          display: 'grid',
          gridTemplate: `"label value" / 1fr fit-content(50%)`,
          margin: 0,
          paddingBlock: theme.spacing.lg,
          ...containerQueryWithFallback(SETTING_CELL_CONTAINER_NAME, SETTING_CELL_CLASS_NAME, stacked, {
            gridTemplate: `"label" "value" / 100%`,
          }),
        }}
      >
        <div css={{ gridArea: 'label' }}>{label}</div>
        <div
          className={SETTING_CELL_VALUE_CLASS_NAME}
          css={{
            gridArea: 'value',
            minWidth: 0,
            ...containerQueryWithFallback(SETTING_CELL_CONTAINER_NAME, SETTING_CELL_VALUE_CLASS_NAME, stacked, {
              marginBlockStart: theme.spacing.sm,
            }),
          }}
        >
          {value}
        </div>
      </Component>
    </div>
  );
};

interface SettingCellControlProps {
  label: ReactNode;
  description?: ReactNode;
  children: (ids: { labelId: string }) => ReactNode;
}

/** A row whose value is a form control; the control is labelled by `labelId`. */
const SettingCellControl = ({ label, description, children }: SettingCellControlProps) => {
  const labelId = useElementId('setting-label');

  return (
    <SettingCellRoot
      label={
        <>
          <label id={labelId}>
            <Typography.Text bold>{label}</Typography.Text>
          </label>
          {description && <Typography.Hint>{description}</Typography.Hint>}
        </>
      }
      value={children({ labelId })}
    />
  );
};

interface SettingCellStaticProps {
  label: ReactNode;
  description?: ReactNode;
  children: ReactNode;
}

/** A read-only row, rendered as its own `<section>` with the label as a definition term. */
const SettingCellStatic = ({ label, description, children }: SettingCellStaticProps) => {
  return (
    <SettingCellRoot
      as="section"
      label={
        <>
          <dfn css={{ fontStyle: 'unset' }}>
            <Typography.Text bold>{label}</Typography.Text>
          </dfn>
          {description && <Typography.Hint>{description}</Typography.Hint>}
        </>
      }
      value={children}
    />
  );
};

export const SettingCell = Object.assign(SettingCellRoot, {
  Control: SettingCellControl,
  Static: SettingCellStatic,
});
