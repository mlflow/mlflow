import {
  t_global_spacer_lg,
  t_global_spacer_md,
  t_global_spacer_sm,
  t_global_spacer_xl,
  t_global_spacer_xs,
  t_global_spacer_2xl,
  t_global_spacer_4xl,
  t_global_spacer_3xl,
} from '@patternfly/react-tokens';
import { convertRemStringToPx } from '../utils';

export const patternflySpacing = {
  xs: convertRemStringToPx(t_global_spacer_xs.value),
  sm: convertRemStringToPx(t_global_spacer_sm.value),
  md: convertRemStringToPx(t_global_spacer_md.value),
  lg: convertRemStringToPx(t_global_spacer_lg.value),
  xl: convertRemStringToPx(t_global_spacer_xl.value),
  '2xl': convertRemStringToPx(t_global_spacer_2xl.value),
  '3xl': convertRemStringToPx(t_global_spacer_3xl.value),
  '4xl': convertRemStringToPx(t_global_spacer_4xl.value),
};
