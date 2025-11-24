// Dark mode colors
const blackAlpha87 = 'rgba(0, 0, 0, 0.87)';
const blackAlpha61 = 'rgba(0, 0, 0, 0.61)';
const blackAlpha45 = 'rgba(0, 0, 0, 0.45)';
const blackAlpha26 = 'rgba(0, 0, 0, 0.26)';
// Light mode colors
const blackAlpha13 = 'rgba(0, 0, 0, 0.13)';
const blackAlpha08 = 'rgba(0, 0, 0, 0.08)';
const blackAlpha05 = 'rgba(0, 0, 0, 0.05)';
const blackAlpha02 = 'rgba(0, 0, 0, 0.02)';
export const getShadows = (isDarkMode) => {
    return isDarkMode
        ? {
            xs: `0px 1px 0px 0px ${blackAlpha45}`,
            sm: ` 0px 2px 3px -1px ${blackAlpha45}, 0px 1px 0px 0px ${blackAlpha26}`,
            md: `0px 3px 6px 0px ${blackAlpha45}`,
            lg: `0px 2px 16px 0px ${blackAlpha61}`,
            xl: `0px 8px 40px 0px ${blackAlpha87}`,
        }
        : {
            xs: `0px 1px 0px 0px ${blackAlpha05}`,
            sm: ` 0px 2px 3px -1px ${blackAlpha05}, 0px 1px 0px 0px ${blackAlpha02}`,
            md: `0px 3px 6px 0px ${blackAlpha05}`,
            lg: `0px 2px 16px 0px ${blackAlpha08}`,
            xl: `0px 8px 40px 0px ${blackAlpha13}`,
        };
};
//# sourceMappingURL=shadows.js.map