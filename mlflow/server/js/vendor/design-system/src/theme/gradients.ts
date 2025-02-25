import { getColors } from './colors';

type DesignSystemGradients = {
  aiBorderGradient: string;
};

export function getGradients(isDarkMode: boolean): DesignSystemGradients {
  const colors = getColors(isDarkMode);
  return {
    aiBorderGradient: `linear-gradient(135deg, ${colors.branded.ai.gradientStart} 20.5%, ${colors.branded.ai.gradientMid} 46.91%, ${colors.branded.ai.gradientEnd} 79.5%)`,
  };
}
