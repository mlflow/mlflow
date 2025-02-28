import type { DesignSystemThemeInterface } from '../../design-system/Hooks/useDesignSystemTheme';

/*
    Used to get output of hooks into ArgTable storybooks.
    Please replace if there is a better way to show a TypeScript interface in Storybook MDX.
 */

export const UseDesignSystemThemeComponent: React.FC<DesignSystemThemeInterface> = (props) => {
  return <span {...props}>Hello</span>;
};
