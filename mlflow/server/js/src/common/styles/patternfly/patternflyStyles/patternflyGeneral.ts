import {
  t_global_border_radius_medium,
  t_global_box_shadow_lg,
  t_global_box_shadow_sm,
  t_global_border_width_regular,
} from '@patternfly/react-tokens';
import { convertPxStringToPx } from '../utils';

export const patternflyGeneral = {
  borderRadiusBase: convertPxStringToPx(t_global_border_radius_medium.value),
  borderWidth: convertPxStringToPx(t_global_border_width_regular.value),
};

export const patternflyShadowVariables = {
  shadowLow: t_global_box_shadow_sm.value,
  shadowHigh: t_global_box_shadow_lg.value,
} as any;
