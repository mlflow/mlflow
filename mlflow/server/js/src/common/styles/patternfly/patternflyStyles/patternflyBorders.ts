import {
  t_global_border_radius_sharp,
  t_global_border_radius_small,
  t_global_border_radius_medium,
  t_global_border_radius_large,
  t_global_border_radius_pill,
} from '@patternfly/react-tokens';
import { convertPxStringToPx } from '../utils';
export const patternflyBorders = {
  borderRadius0: convertPxStringToPx(t_global_border_radius_sharp.value),
  borderRadiusSm: convertPxStringToPx(t_global_border_radius_small.value),
  borderRadiusMd: convertPxStringToPx(t_global_border_radius_medium.value),
  borderRadiusLg: convertPxStringToPx(t_global_border_radius_large.value),
  borderRadiusFull: convertPxStringToPx(t_global_border_radius_pill.value),
};

export const patternflyLegacyBorders = {
  borderRadiusMd: convertPxStringToPx(t_global_border_radius_medium.value),
  borderRadiusLg: convertPxStringToPx(t_global_border_radius_large.value),
};
