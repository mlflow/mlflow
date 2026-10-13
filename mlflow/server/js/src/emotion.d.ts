import '@emotion/react';
import type { DesignSystemThemeInterface } from '@databricks/design-system';

type ThemeType = DesignSystemThemeInterface['theme'];

declare module '@emotion/react' {
  export interface Theme extends ThemeType {}
}
