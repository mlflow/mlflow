import type { TreeDataNode } from '@databricks/design-system';

export const SPAN_NAMES_MIN_WIDTH = 100;
export const SPAN_NAMES_WITHOUT_GANTT_MIN_WIDTH = 280;
export const GANTT_BARS_MIN_WIDTH = 200;

export interface TimelineTreeNode extends Pick<TreeDataNode, 'key' | 'title' | 'icon'> {
  start: number;
  end: number;
  children?: TimelineTreeNode[];
}

export type HierarchyBar = {
  shouldRender: boolean;
  // the bar will render blue if this is true
  isActive: boolean;
};
