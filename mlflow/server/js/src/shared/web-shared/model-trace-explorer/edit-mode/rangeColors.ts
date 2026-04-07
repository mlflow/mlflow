// Fixed palette of range colors, cycled by range index.
// Each entry has a primary color for borders/badges and a background tint.
export const RANGE_COLORS = [
  { primary: '#3b82f6', background: 'rgba(59, 130, 246, 0.08)' }, // blue
  { primary: '#8b5cf6', background: 'rgba(139, 92, 246, 0.08)' }, // purple
  { primary: '#10b981', background: 'rgba(16, 185, 129, 0.08)' }, // green
  { primary: '#f59e0b', background: 'rgba(245, 158, 11, 0.08)' }, // amber
  { primary: '#ef4444', background: 'rgba(239, 68, 68, 0.08)' }, // red
  { primary: '#06b6d4', background: 'rgba(6, 182, 212, 0.08)' }, // cyan
] as const;

export const getRangeColor = (index: number) => RANGE_COLORS[index % RANGE_COLORS.length];
