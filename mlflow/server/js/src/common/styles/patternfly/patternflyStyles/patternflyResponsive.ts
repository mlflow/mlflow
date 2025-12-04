// PatternFly responsive breakpoints override for Databricks Design System
// PatternFly uses similar breakpoints but with slightly different values

import {
  t_global_breakpoint_md,
  t_global_breakpoint_sm,
  t_global_breakpoint_xs,
  t_global_breakpoint_2xl,
} from '@patternfly/react-tokens';
import { t_global_breakpoint_lg } from '@patternfly/react-tokens';
import { t_global_breakpoint_xl } from '@patternfly/react-tokens';
import { convertPxStringToPx } from '../utils';

type availableBreakpoints = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
type ResponsiveOptions<RType> = Record<availableBreakpoints, RType>;

const breakpoints: ResponsiveOptions<number> = {
  xs: convertPxStringToPx(t_global_breakpoint_xs.value),
  sm: convertPxStringToPx(t_global_breakpoint_sm.value),
  md: convertPxStringToPx(t_global_breakpoint_md.value),
  lg: convertPxStringToPx(t_global_breakpoint_lg.value),
  xl: convertPxStringToPx(t_global_breakpoint_xl.value),
  xxl: convertPxStringToPx(t_global_breakpoint_2xl.value),
};

const mediaQueries: ResponsiveOptions<string> = {
  xs: `@media (max-width: ${breakpoints.xs}px)`,
  sm: `@media (min-width: ${breakpoints.sm}px)`,
  md: `@media (min-width: ${breakpoints.md}px)`,
  lg: `@media (min-width: ${breakpoints.lg}px)`,
  xl: `@media (min-width: ${breakpoints.xl}px)`,
  xxl: `@media (min-width: ${breakpoints.xxl}px)`,
};

export const patternflyResponsive = { breakpoints, mediaQueries };
