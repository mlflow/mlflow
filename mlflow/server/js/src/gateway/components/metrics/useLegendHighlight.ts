import { useState, useCallback } from 'react';

export function useLegendHighlight(defaultOpacity = 1, dimmedOpacity = 0.2) {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  const getOpacity = useCallback(
    (itemKey: string) => {
      if (hoveredItem === null) return defaultOpacity;
      return hoveredItem === itemKey ? defaultOpacity : dimmedOpacity;
    },
    [hoveredItem, defaultOpacity, dimmedOpacity],
  );

  const handleLegendMouseEnter = useCallback((data: { value: string }) => {
    setHoveredItem(data.value);
  }, []);

  const handleLegendMouseLeave = useCallback(() => {
    setHoveredItem(null);
  }, []);

  return {
    hoveredItem,
    getOpacity,
    handleLegendMouseEnter,
    handleLegendMouseLeave,
  };
}
