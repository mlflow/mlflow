import { Theme } from '@emotion/react';
import { colorList } from './patternflyStyles/patternflyColors';
import { patternflySpacing } from './patternflyStyles/patternflySpacing';
import { patternflyGeneral, patternflyShadowVariables } from './patternflyStyles/patternflyGeneral';
import { patternflyTypography } from './patternflyStyles/patternflyTypography';
import { patternflyBorders, patternflyLegacyBorders } from './patternflyStyles/patternflyBorders';
import { patternflyResponsive } from './patternflyStyles/patternflyResponsive';
import { patternflyBoxShadows } from './patternflyStyles/patternflyShadows';

export const PATTERN_FLY_TOKEN_TRANSLATION = (theme: Theme): Theme => ({
  ...theme,
  colors: {
    ...theme.colors,
    ...colorList,
  },
  spacing: {
    ...theme.spacing,
    ...patternflySpacing,
  },
  general: {
    ...theme.general,
    ...patternflyGeneral,
    ...patternflyShadowVariables,
  },
  typography: {
    ...theme.typography,
    ...patternflyTypography,
  },
  shadows: {
    ...theme.shadows,
    ...patternflyBoxShadows,
  },
  borders: {
    ...theme.borders,
    ...patternflyBorders,
  },
  legacyBorders: {
    ...theme.legacyBorders,
    ...patternflyLegacyBorders,
  },
  responsive: {
    ...theme.responsive,
    ...patternflyResponsive,
  },
});
