import type { ComponentType } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  CheckCircleIcon,
  ChecklistIcon,
  ClockIcon,
  HashIcon,
  Typography,
  useDesignSystemTheme,
  WrenchIcon,
  XCircleFillIcon,
} from '@databricks/design-system';

const ICON_NAMES = ['wrench', 'clock', 'checkCircle', 'xCircle', 'hash', 'checklist'] as const;
type IconName = (typeof ICON_NAMES)[number];

const TONES = ['info', 'success', 'warning', 'danger'] as const;
type Tone = (typeof TONES)[number];

const ICON_BY_NAME: Record<IconName, ComponentType> = {
  wrench: WrenchIcon,
  clock: ClockIcon,
  checkCircle: CheckCircleIcon,
  xCircle: XCircleFillIcon,
  hash: HashIcon,
  checklist: ChecklistIcon,
};

/**
 * Schema (API) for the custom StatCard component. Mirrors the props of the
 * StatCard used on the experiment tool-calls overview page, but exposes a
 * `tone` enum instead of raw colors so values stay theme-driven.
 */
export const StatCardApi = {
  name: 'StatCard',
  schema: z
    .object({
      value: DynamicStringSchema.describe('The metric value to display, e.g. "14" or "275.71ms".'),
      label: DynamicStringSchema.describe('The caption describing the metric.'),
      icon: z.enum(ICON_NAMES).describe('The icon to display next to the value.').optional(),
      tone: z
        .enum(TONES)
        .default('info')
        .describe('The color tone applied to the icon (info/success/warning/danger).')
        .optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const StatCard = createComponentImplementation(StatCardApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const value = typeof props.value === 'string' ? props.value : String(props.value ?? '');
  const label = typeof props.label === 'string' ? props.label : String(props.label ?? '');
  const tone: Tone = (props.tone as Tone) ?? 'info';
  const iconName: IconName = (props.icon as IconName) ?? 'wrench';

  const toneColor: Record<Tone, string> = {
    info: theme.colors.blue500,
    success: theme.colors.green500,
    warning: theme.colors.yellow500,
    danger: theme.colors.red500,
  };

  const iconColor = toneColor[tone];
  const iconBgColor = `${iconColor}1A`;
  const IconComponent = ICON_BY_NAME[iconName];

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        flex: 1,
        minWidth: 160,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: theme.general.iconSize,
          height: theme.general.iconSize,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: iconBgColor,
          color: iconColor,
          flexShrink: 0,
        }}
      >
        <IconComponent />
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <Typography.Text bold size="lg">
          {value}
        </Typography.Text>
        <Typography.Text color="secondary" size="sm">
          {label}
        </Typography.Text>
      </div>
    </div>
  );
});
