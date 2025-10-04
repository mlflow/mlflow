import type { ComponentType, ReactNode } from 'react';

export type HomeNavigationLink =
  | { type: 'internal'; to: string }
  | { type: 'external'; href: string; target?: '_blank'; rel?: string };

export interface HomeQuickActionDefinition {
  id: string;
  icon: ComponentType<{ className?: string; css?: any }>;
  title: ReactNode;
  description: ReactNode;
  link: HomeNavigationLink;
  componentId: string;
}

export interface HomeNewsItemDefinition {
  id: string;
  title: ReactNode;
  description: ReactNode;
  link: HomeNavigationLink;
  componentId: string;
  thumbnail: {
    gradient: string;
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
