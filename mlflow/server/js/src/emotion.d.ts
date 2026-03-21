import '@emotion/react';
import type { DesignSystemThemeInterface } from '@databricks/design-system';

type ThemeType = DesignSystemThemeInterface['theme'];

declare module '@emotion/react' {
  // eslint-disable-next-line @typescript-eslint/no-empty-interface
  export interface Theme extends ThemeType {}
}
