import '@emotion/react';
import type { Theme as DuBoisTheme } from './theme';

declare module '@emotion/react' {
  export interface Theme extends DuBoisTheme {}
}
