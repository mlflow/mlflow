import { useCallback, useState } from 'react';
import type { SpanRange, SpanSelector, TraceView } from './useTraceViews';

type DraftView = Pick<TraceView, 'view_id' | 'name' | 'ranges'>;

export const useTraceViewEditMode = () => {
  const [draftView, setDraftView] = useState<DraftView | null>(null);

  const isEditMode = draftView !== null;

  const enterEditMode = useCallback((existingView?: TraceView) => {
    if (existingView) {
      setDraftView({
        view_id: existingView.view_id,
        name: existingView.name,
        ranges: existingView.ranges.map((r) => ({ ...r })),
      });
    } else {
      setDraftView({ view_id: '', name: '', ranges: [] });
    }
  }, []);

  const exitEditMode = useCallback(() => {
    setDraftView(null);
  }, []);

  const setName = useCallback((name: string) => {
    setDraftView((prev) => (prev ? { ...prev, name } : prev));
  }, []);

  const addRange = useCallback((from: SpanSelector, to?: SpanSelector) => {
    setDraftView((prev) => {
      if (!prev) return prev;
      const position = prev.ranges.length;
      const newRange: SpanRange = {
        from_selector: from,
        to_selector: to ?? null,
        label: `Range ${position + 1}`,
        description: '',
        position,
      };
      return { ...prev, ranges: [...prev.ranges, newRange] };
    });
  }, []);

  const removeRange = useCallback((index: number) => {
    setDraftView((prev) => {
      if (!prev) return prev;
      const ranges = prev.ranges
        .filter((_, i) => i !== index)
        .map((r, i) => ({ ...r, position: i }));
      return { ...prev, ranges };
    });
  }, []);

  const updateRange = useCallback((index: number, updates: Partial<SpanRange>) => {
    setDraftView((prev) => {
      if (!prev) return prev;
      const ranges = prev.ranges.map((r, i) => (i === index ? { ...r, ...updates } : r));
      return { ...prev, ranges };
    });
  }, []);

  const reorderRanges = useCallback((fromIndex: number, toIndex: number) => {
    setDraftView((prev) => {
      if (!prev) return prev;
      const ranges = [...prev.ranges];
      const [moved] = ranges.splice(fromIndex, 1);
      ranges.splice(toIndex, 0, moved);
      return { ...prev, ranges: ranges.map((r, i) => ({ ...r, position: i })) };
    });
  }, []);

  return {
    isEditMode,
    draftView,
    enterEditMode,
    exitEditMode,
    setName,
    addRange,
    removeRange,
    updateRange,
    reorderRanges,
  };
};
