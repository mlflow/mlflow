import '@emotion/react';
import { DesignSystemThemeInterface } from '@databricks/design-system';

type ThemeType = DesignSystemThemeInterface['theme'];

declare module '@emotion/react' {
  // eslint-disable-next-line @typescript-eslint/no-empty-interface, @typescript-eslint/no-empty-object-type
  export interface Theme extends ThemeType {}
}
