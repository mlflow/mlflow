import { Button, PlusCircleBorderIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { useCallback, useEffect, useRef } from 'react';
import { FormattedMessage } from 'react-intl';

/**
 * Width of the CTA column
 */
const COLUMN_CTA_WIDTH = 180;

/**
 * CSS classes used internally
 */
const CLASS_OUT_OF_VIEWPORT = 'is-out-of-viewport';
const CLASS_IS_HIDDEN = 'is-hidden';
const CLASS_IS_MINIMIZED = 'is-minimized';

/**
 * List all necessary agGrid sub-element classes
 */
const AG_GRID_CLS = {
  ROOT: '.ag-root',
  LEFT_COLS_CONTAINER: '.ag-pinned-left-cols-container',
  COLS_CONTAINER: '.ag-center-cols-container',
  HEADER: '.ag-header',
};

interface ExperimentViewRunsTableAddColumnCTAProps {
  onClick: () => void;
  gridContainerElement: HTMLElement | null;
  isInitialized: boolean;
  visible?: boolean;
  moreAvailableParamsAndMetricsColumnCount?: number;
}

/**
 * Component displaying dynamic table column with "add metrics and parameters" CTA.
 *
 * Sample usage:
 *
 * const [gridInitialized, setGridInitialized] = useState(false);
 *
 * return (
 *   <div ref={containerElement}>
 *     <AgGrid onGridReady={() => setGridInitialized(true)} {...} />
 *     <ExperimentViewRunsTableAddColumnCTA
 *       gridContainerElement={containerElement.current}
 *       isInitialized={gridInitialized}
 *       onAddColumnClicked={onAddColumnClicked}
 *       visible={!isLoading}
 *       moreAvailableParamsAndMetricsColumnCount={3}
 *     />
 *   </div>
 * );
 */
export const ExperimentViewRunsTableAddColumnCTA = ({
  onClick,
  gridContainerElement,
  isInitialized,
  visible,
  moreAvailableParamsAndMetricsColumnCount = 0,
}: ExperimentViewRunsTableAddColumnCTAProps) => {
  const ctaRef = useRef<HTMLDivElement>(null);

  const savedContainerRef = useRef<HTMLElement>();

  const initialize = useCallback((containerElement: HTMLElement) => {
    if (!ctaRef.current || !window.ResizeObserver || !containerElement) {
      return undefined;
    }

    const targetElement = ctaRef.current;

    /**
     * On initialization, first gather all the agGrid sub-elements
     */
    const rootElement = containerElement.querySelector(AG_GRID_CLS.ROOT);
    const refLeftElem = containerElement.querySelector(AG_GRID_CLS.LEFT_COLS_CONTAINER);
    const refCenterElem = containerElement.querySelector(AG_GRID_CLS.COLS_CONTAINER);
    const refHeaderElem = containerElement.querySelector(AG_GRID_CLS.HEADER);

    /**
     * Initialize variables used for position calculation
     */
    let gridAreaWidth = 0;
    let leftColContainerWidth = 0;
    let centerColContainerWidth = 0;
    let colContainerHeight = 0;
    let headerHeight = 0;

    /**
     * Execute only if all elements are in place
     */
    if (refLeftElem && refCenterElem && refHeaderElem && rootElement) {
      /**
       * Hook up an resize observer
       */
      const resizeObserver = new ResizeObserver((entries) => {
        /**
         * For every changed element, gather the exact dimensions
         */
        for (const entry of entries) {
          if (entry.target === rootElement) {
            gridAreaWidth = entry.contentRect.width;
          }
          if (entry.target === refLeftElem) {
            leftColContainerWidth = entry.contentRect.width;
            colContainerHeight = entry.contentRect.height;
          }
          if (entry.target === refHeaderElem) {
            headerHeight = entry.contentRect.height;
          }
          if (entry.target === refCenterElem) {
            centerColContainerWidth = entry.contentRect.width;
          }
        }

        /**
         * Our "left" position will be offset by column container widths
         */
        const calculatedLeft = leftColContainerWidth + centerColContainerWidth;

        /**
         * Our "top"  position will be offset by the header height
         */
        const calculatedTop = headerHeight;

        /**
         * If the column is out of viewport (expanding out of the root element),
         * add proper CSS class to hide it
         */
        const isOutOfViewport = calculatedLeft + COLUMN_CTA_WIDTH >= gridAreaWidth;
        isOutOfViewport
          ? savedContainerRef.current?.classList.add(CLASS_OUT_OF_VIEWPORT)
          : savedContainerRef.current?.classList.remove(CLASS_OUT_OF_VIEWPORT);

        /**
         * If the available height is too low, add a class that indicates
         * that we should display minimized version
         */
        const shouldBeMinimized = colContainerHeight < 100;
        shouldBeMinimized
          ? savedContainerRef.current?.classList.add(CLASS_IS_MINIMIZED)
          : savedContainerRef.current?.classList.remove(CLASS_IS_MINIMIZED);

        /**
         * Finally, set proper values as CSS transform property. Use 3d transform
         * to ensure hardware acceleration.
         */
        targetElement.style.transform = `translate3d(${calculatedLeft}px, ${calculatedTop}px, 0)`;

        /**
         * Set target height and add 1px to accomodate the border.
         */
        targetElement.style.height = `${colContainerHeight + 1}px`;
      });

      /**
       * Setup observer with all the necessary elements.
       */
      resizeObserver.observe(refLeftElem);
      resizeObserver.observe(refCenterElem);
      resizeObserver.observe(refHeaderElem);
      resizeObserver.observe(rootElement);

      /**
       * After cleanup, disconnect the observer.
       */
      return () => resizeObserver.disconnect();
    }
    return undefined;
  }, []);

  useEffect(() => {
    if (isInitialized && gridContainerElement) {
      savedContainerRef.current = gridContainerElement;
      initialize(gridContainerElement);
    }
  }, [initialize, isInitialized, gridContainerElement]);

  /**
   * This component works only if ResizeObserver is supported by the browser.
   * If it's not supported, return nothing.
   */
  if (!window.ResizeObserver) {
    return null;
  }

  return (
    <div ref={ctaRef} css={styles.columnContainer} className={visible ? '' : CLASS_IS_HIDDEN}>
      {visible && (
        <div css={styles.buttonContainer}>
          <Button css={styles.button} type='link' onClick={onClick}>
            <PlusCircleBorderIcon css={styles.buttonIcon} />
            <div css={styles.caption}>
              <FormattedMessage
                defaultMessage='Show more metrics and parameters'
                description='Label for a CTA button in experiment runs table which invokes column management dropdown'
              />{' '}
              {moreAvailableParamsAndMetricsColumnCount ? (
                <>({moreAvailableParamsAndMetricsColumnCount})</>
              ) : null}
            </div>
          </Button>
        </div>
      )}
    </div>
  );
};

const styles = {
  columnContainer: (theme: Theme) => ({
    width: COLUMN_CTA_WIDTH,
    height: 0,
    position: 'absolute' as const,
    border: `1px solid ${theme.colors.border}`,
    borderLeft: 0,
    borderTop: 0,
    top: 0,
    left: 0,
    // Slightly expanded icon to conform to the designs
    svg: {
      height: 16,
      width: 16,
    },
    willChange: 'transform' as const,
    transform: 'translate3d(0, 0, 0)',
    [`.${CLASS_IS_MINIMIZED} &`]: {
      display: 'flex',
      alignItems: 'center' as const,
    },
    [`&.${CLASS_IS_HIDDEN}, .${CLASS_OUT_OF_VIEWPORT} &`]: {
      display: 'none',
    },
  }),
  buttonContainer: (theme: Theme) => ({
    position: 'sticky' as const,
    top: 0,
    paddingLeft: theme.spacing.lg,
    paddingRight: theme.spacing.lg,
    paddingTop: theme.spacing.lg,
    width: '100%',
    [`.${CLASS_IS_MINIMIZED} &`]: {
      paddingTop: theme.spacing.xs,
    },
  }),
  button: { whiteSpace: 'pre-wrap' as const, width: '100%' },
  buttonIcon: (theme: Theme) => ({ color: theme.colors.textSecondary }),
  caption: (theme: Theme) => ({
    color: theme.colors.textSecondary,
    [`.${CLASS_IS_MINIMIZED} &`]: {
      display: 'none',
    },
  }),
};
