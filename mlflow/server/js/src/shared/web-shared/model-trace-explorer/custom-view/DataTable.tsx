import type { ComponentType } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  ChecklistIcon,
  ClockIcon,
  HashIcon,
  ListBorderIcon,
  Typography,
  useDesignSystemTheme,
  WrenchIcon,
} from '@databricks/design-system';

const ICON_NAMES = ['list', 'wrench', 'clock', 'hash', 'checklist'] as const;
type IconName = (typeof ICON_NAMES)[number];

const ALIGNMENTS = ['left', 'center', 'right'] as const;
type Alignment = (typeof ALIGNMENTS)[number];

const ICON_BY_NAME: Record<IconName, ComponentType> = {
  list: ListBorderIcon,
  wrench: WrenchIcon,
  clock: ClockIcon,
  hash: HashIcon,
  checklist: ChecklistIcon,
};

/**
 * Schema (API) for the generic DataTable component. It renders an optional
 * titled header followed by a column-aligned table. It is intentionally
 * domain-agnostic: callers describe `columns` and supply `rows` whose `cells`
 * are positional (aligned to `columns` by index). An optional per-row `color`
 * renders a leading indicator dot in the first cell, mirroring the summary
 * tables on the experiment overview page.
 */
export const DataTableApi = {
  name: 'DataTable',
  schema: z
    .object({
      title: DynamicStringSchema.describe('Optional heading shown above the table.').optional(),
      icon: z.enum(ICON_NAMES).describe('Optional icon shown next to the title.').optional(),
      columns: z
        .array(
          z.object({
            label: DynamicStringSchema.describe('The column header text.'),
            align: z.enum(ALIGNMENTS).describe('Horizontal alignment of the column.').default('left').optional(),
          }),
        )
        .describe('The ordered column definitions.'),
      rows: z
        .array(
          z.object({
            color: z.string().describe('Optional CSS color for a leading indicator dot.').optional(),
            cells: z.array(DynamicStringSchema).describe('Cell values, positionally aligned to `columns`.'),
          }),
        )
        .describe('The table rows.'),
      emptyMessage: DynamicStringSchema.describe('Text shown when there are no rows.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

export const DataTable = createComponentImplementation(DataTableApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const columns = Array.isArray(props.columns) ? props.columns : [];
  const rows = Array.isArray(props.rows) ? props.rows : [];
  const title = props.title ? asString(props.title) : undefined;
  const emptyMessage = props.emptyMessage ? asString(props.emptyMessage) : 'No data to display.';
  const IconComponent = props.icon ? ICON_BY_NAME[props.icon as IconName] : undefined;

  // First column is wider (it usually holds the row name); the rest share evenly.
  const gridTemplateColumns =
    columns.length > 0 ? `minmax(120px, 2fr) ${'1fr '.repeat(Math.max(columns.length - 1, 0)).trim()}`.trim() : '1fr';

  const rowGridStyle = {
    display: 'grid',
    gridTemplateColumns,
    gap: theme.spacing.lg,
    alignItems: 'center',
  } as const;

  const alignToJustify: Record<Alignment, string> = {
    left: 'flex-start',
    center: 'center',
    right: 'flex-end',
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {title && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {IconComponent && (
            <span css={{ display: 'flex', color: theme.colors.textSecondary }}>
              <IconComponent />
            </span>
          )}
          <Typography.Text bold size="lg">
            {title}
          </Typography.Text>
        </div>
      )}

      {rows.length === 0 ? (
        <Typography.Text color="secondary">{emptyMessage}</Typography.Text>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          <div
            css={{
              ...rowGridStyle,
              padding: `${theme.spacing.sm}px 0`,
              borderBottom: `1px solid ${theme.colors.border}`,
            }}
          >
            {columns.map((column, columnIndex) => {
              const align: Alignment = (column?.align as Alignment) ?? 'left';
              return (
                <div
                  key={columnIndex}
                  css={{
                    display: 'flex',
                    justifyContent: alignToJustify[align],
                    color: theme.colors.textSecondary,
                    fontSize: theme.typography.fontSizeSm,
                    fontWeight: 600,
                  }}
                >
                  {asString(column?.label)}
                </div>
              );
            })}
          </div>

          <div css={{ maxHeight: 240, overflowY: 'auto' }}>
            {rows.map((row, rowIndex) => {
              const cells = Array.isArray(row?.cells) ? row.cells : [];
              return (
                <div
                  key={rowIndex}
                  css={{
                    ...rowGridStyle,
                    padding: `${theme.spacing.md}px 0`,
                    borderBottom: `1px solid ${theme.colors.border}`,
                    '&:last-child': { borderBottom: 'none' },
                  }}
                >
                  {columns.map((column, columnIndex) => {
                    const align: Alignment = (column?.align as Alignment) ?? 'left';
                    const cellValue = asString(cells[columnIndex]);
                    const isFirstColumn = columnIndex === 0;

                    if (isFirstColumn) {
                      return (
                        <div
                          key={columnIndex}
                          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}
                        >
                          {row?.color && (
                            <span
                              css={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                backgroundColor: row.color,
                                flexShrink: 0,
                              }}
                            />
                          )}
                          <Typography.Text
                            css={{
                              fontFamily: 'monospace',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {cellValue}
                          </Typography.Text>
                        </div>
                      );
                    }

                    return (
                      <div key={columnIndex} css={{ display: 'flex', justifyContent: alignToJustify[align] }}>
                        <Typography.Text color="secondary">{cellValue}</Typography.Text>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
});
