import { useEffect, useMemo } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { create } from '@databricks/web-shared/zustand';

interface FloatingObstructionStore {
  /** id -> width (px) reserved on the right edge by an open surface (e.g. a drawer). */
  obstructions: Record<string, number>;
  setObstruction: (id: string, reservedRightPx: number) => void;
  removeObstruction: (id: string) => void;
}

const useFloatingObstructionStore = create<FloatingObstructionStore>((set) => ({
  obstructions: {},
  setObstruction: (id, reservedRightPx) =>
    set((state) => ({ obstructions: { ...state.obstructions, [id]: reservedRightPx } })),
  removeObstruction: (id) =>
    set((state) => {
      if (!(id in state.obstructions)) {
        return state;
      }
      const next = { ...state.obstructions };
      delete next[id];
      return { obstructions: next };
    }),
}));

/**
 * Register that some surface (e.g. a right-side drawer) is currently occupying the right
 * edge of the viewport, so the floating Assistant button can move clear of it instead of
 * overlapping content. Pass the width (px) reserved on the right while the surface is open,
 * or 0 when it isn't. Multiple surfaces compose; the button uses the widest reservation.
 */
export const useRegisterFloatingObstruction = (reservedRightPx: number) => {
  const id = useMemo(() => uuidv4(), []);
  const setObstruction = useFloatingObstructionStore((state) => state.setObstruction);
  const removeObstruction = useFloatingObstructionStore((state) => state.removeObstruction);
  useEffect(() => {
    if (!reservedRightPx || reservedRightPx <= 0) {
      removeObstruction(id);
      return undefined;
    }
    setObstruction(id, reservedRightPx);
    return () => removeObstruction(id);
  }, [id, reservedRightPx, setObstruction, removeObstruction]);
};

/** The widest reservation (px) currently held on the right edge, or 0 if none. */
export const useFloatingObstructionWidth = (): number =>
  useFloatingObstructionStore((state) => {
    const widths = Object.values(state.obstructions);
    return widths.length > 0 ? Math.max(...widths) : 0;
  });
