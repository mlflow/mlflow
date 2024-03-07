import { Button, ChevronLeftIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import { throttle } from 'lodash';
import React, { useEffect, useMemo, useRef, useState } from 'react';

// In milliseconds
const SCROLL_THROTTLE = 100;

/**
 * A horizontal carousel component that allows scrolling through a list of elements.
 */
export const HorizontalCarousel = ({ children, title }: { children: React.ReactNode; title: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();

  const gapBetweenCards = theme.spacing.md;

  // Determines the width of a single element in the list.
  // Assumes that all elements have the same width.
  const [elementWidth, setElementWidth] = useState(0);
  const elementWidthWithGap = elementWidth + gapBetweenCards;

  const listWrapper = useRef<HTMLDivElement>(null);

  // Determines if the list is overflowing, i.e. if we should show the scroll buttons
  const [isOverflowing, setOverflowing] = useState(false);

  // Determines the width of a single page when using "next"/"previous" buttons
  const [pageWidth, setPageWidth] = useState(0);

  // Determines the index of the leftmost visible element
  const [leftmostElement, setLeftmostElement] = useState(0);

  // Determines if the scroll buttons should be disabled
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  // Throttled scroll handler for the list
  const scrollHandler = useMemo(
    () =>
      throttle((target: Element) => {
        const scrollValue = target.scrollLeft;
        const scrollWidth = target.scrollWidth;
        const visibleIndex = Math.floor(scrollValue / elementWidthWithGap);
        setCanScrollLeft(scrollValue > 0);
        setCanScrollRight(scrollValue + target.clientWidth < scrollWidth);
        setLeftmostElement(visibleIndex);
      }, SCROLL_THROTTLE),
    [elementWidthWithGap],
  );

  // Set up resize observer to determine if the list is overflowing
  useEffect(() => {
    const resizeObserver = new ResizeObserver(([{ target }]) => {
      const { scrollWidth, clientWidth } = target;
      scrollHandler(target);
      setElementWidth(target.firstChild instanceof HTMLElement ? target.firstChild.offsetWidth : 0);
      setPageWidth(clientWidth);
      setOverflowing(scrollWidth > clientWidth);
    });
    resizeObserver.observe(listWrapper.current as HTMLDivElement);
    return () => {
      resizeObserver.disconnect();
    };
  }, [scrollHandler]);

  const scrollTo = (elementOffset: number) => {
    if (listWrapper.current) {
      listWrapper.current.scrollTo({
        left: elementOffset * elementWidthWithGap,
        behavior: 'smooth',
      });
    }
  };

  // Scroll handlers for the buttons
  const scrollLeft = () => scrollTo(leftmostElement - Math.round(pageWidth / elementWidthWithGap));
  const scrollRight = () => scrollTo(leftmostElement + Math.round(pageWidth / elementWidthWithGap));

  return (
    <>
      {/* List header */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: theme.spacing.xs,
          alignItems: 'center',
        }}
      >
        <div>{title}</div>
        <div css={{ height: theme.spacing.lg }}>
          {isOverflowing && (
            <>
              <Button
                componentId="codegen_mlflow_app_src_common_components_horizontalcarousel.tsx_93"
                size="small"
                disabled={!canScrollLeft}
                onClick={scrollLeft}
                type="tertiary"
                icon={<ChevronLeftIcon />}
              />
              <Button
                componentId="codegen_mlflow_app_src_common_components_horizontalcarousel.tsx_100"
                size="small"
                disabled={!canScrollRight}
                onClick={scrollRight}
                type="tertiary"
                icon={<ChevronRightIcon />}
              />
            </>
          )}
        </div>
      </div>
      {/* A list element containing cards */}
      <div
        role="list"
        css={{
          display: 'flex',
          gap: gapBetweenCards,
          overflowX: 'auto',
          position: 'relative',
        }}
        ref={listWrapper}
        onScroll={(e) => {
          if (e.target instanceof Element) {
            scrollHandler(e.target);
          }
        }}
      >
        {children}
      </div>
    </>
  );
};
