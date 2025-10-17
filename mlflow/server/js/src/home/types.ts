import type { ComponentType, ReactNode } from 'react';

export interface HomeQuickActionDefinition {
  id: string;
  icon: ComponentType<{ className?: string; css?: any }>;
  title: ReactNode;
  description: ReactNode;
  link: string;
  componentId: string;
}

export interface HomeNewsItemDefinition {
  id: string;
  title: ReactNode;
  description: ReactNode;
  link: string;
  componentId: string;
  thumbnail: {
    gradient: {
      light: string;
      dark: string;
    };
    label?: ReactNode;
    icon?: ComponentType<{ className?: string; css?: any }>;
  };
}

export type HomeExperimentRow = {
  experimentId: string;
  name: string;
  creationTime: number;
  lastUpdateTime: number;
  description?: string;
};
