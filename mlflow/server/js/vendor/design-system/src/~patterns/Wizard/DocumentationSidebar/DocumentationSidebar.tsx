import { isUndefined, noop } from 'lodash';
import type { PropsWithChildren } from 'react';
import React, { useMemo, useState } from 'react';

import type { TooltipProps } from '../../../design-system';
import {
  Button,
  CloseIcon,
  InfoIcon,
  Modal,
  Sidebar,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '../../../design-system';

export type RootProps = PropsWithChildren<{
  /**
   * Initial content id to be displayed
   *
   * @default `undefined`
   */
  initialContentId?: string | undefined;
}>;

export function Root({ children, initialContentId }: RootProps) {
  const [currentContentId, setCurrentContentId] = useState<string | undefined>(initialContentId);

  return (
    <DocumentationSideBarContext.Provider
      value={useMemo(() => ({ currentContentId, setCurrentContentId }), [currentContentId, setCurrentContentId])}
    >
      {children}
    </DocumentationSideBarContext.Provider>
  );
}

export interface DocumentationSideBarState {
  currentContentId: string | undefined;
  setCurrentContentId: (value: string | undefined) => void;
}

const DocumentationSideBarContext = React.createContext<DocumentationSideBarState>({
  currentContentId: undefined,
  setCurrentContentId: noop,
});

export const useDocumentationSidebarContext = (): DocumentationSideBarState => {
  const context = React.useContext(DocumentationSideBarContext);
  return context;
};

export interface TriggerProps<T extends string> extends Omit<TooltipProps, 'content' | 'children'> {
  /**
   * ContentId that will be passed along to the Content.
   */
  contentId: T;

  /**
   * aria-label for the info icon button
   */
  label?: string;

  /**
   * Optional content for tooltip that will wrap the trigger content
   */
  tooltipContent?: React.ReactNode;

  /**
   * Set to true if you want to render your own trigger; requires children to a valid single React node
   *
   * Will default to render an `InfoIcon` wrapped in a button
   */
  asChild?: boolean;

  /**
   * Will be rendered as trigger content if `asChild` is true
   *
   * The node must be an interactive element ex: <Button/>
   */
  children?: React.ReactNode;
}

export function Trigger<T extends string>({
  contentId,
  label,
  tooltipContent,
  asChild,
  children,
  ...tooltipProps
}: TriggerProps<T>) {
  const { theme } = useDesignSystemTheme();
  const { setCurrentContentId } = useDocumentationSidebarContext();
  const triggerProps = useMemo(
    () => ({
      onClick: () => setCurrentContentId(contentId),
      [`aria-label`]: label,
    }),
    [contentId, label, setCurrentContentId],
  );
  const renderAsChild = asChild && React.isValidElement(children);

  return (
    <Tooltip {...tooltipProps} content={tooltipContent}>
      {renderAsChild ? (
        React.cloneElement(children as React.ReactElement, triggerProps)
      ) : (
        <button
          css={{
            border: 'none',
            backgroundColor: 'transparent',
            padding: 0,
            display: 'flex',
            height: 'var(--spacing-md)',
            alignItems: 'center',
            cursor: 'pointer',
          }}
          {...triggerProps}
        >
          <InfoIcon css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }} />
        </button>
      )}
    </Tooltip>
  );
}

export interface ContentChildProps<T extends string> {
  contentId: T;
}

export type ContentProps = {
  /**
   * @default 100%
   */
  width?: number;

  /**
   * This must be a single React element that takes in an optional `contentId: string` prop
   */
  children: React.ReactNode;

  /**
   * Title displayed atop for all content id
   */
  title: string;

  /**
   * The compact modal title; defaults to `title`
   */
  modalTitleWhenCompact?: string;

  /**
   * aria-label for the close button and a button label for the compact modal version
   */
  closeLabel: string;

  /**
   * If true the documentation content will display in a modal instead of a sidebar
   *
   * Example set to true for a specific breakpoint:
   * const displayModalWhenCompact = useMediaQuery({query: `(max-width: ${theme.responsive.breakpoints.lg }px)`})
   */
  displayModalWhenCompact: boolean;
};

export function Content<T extends string>({
  title,
  modalTitleWhenCompact,
  width,
  children,
  closeLabel,
  displayModalWhenCompact,
}: ContentProps) {
  const { theme } = useDesignSystemTheme();
  const { currentContentId, setCurrentContentId } = useDocumentationSidebarContext();

  if (isUndefined(currentContentId)) {
    return null;
  }

  const content = React.isValidElement<ContentChildProps<T>>(children)
    ? React.cloneElement(children, { contentId: currentContentId as T })
    : children;

  if (displayModalWhenCompact) {
    return (
      <Modal
        componentId={`documentation-side-bar-compact-modal-${currentContentId}`}
        visible
        size="wide"
        onOk={() => setCurrentContentId(undefined)}
        okText={closeLabel}
        okButtonProps={{ type: undefined }}
        onCancel={() => setCurrentContentId(undefined)}
        title={modalTitleWhenCompact ?? title}
      >
        {content}
      </Modal>
    );
  }

  return (
    <Sidebar position="right" dangerouslyAppendEmotionCSS={{ border: 'none' }}>
      <Sidebar.Content
        componentId={`documentation-side-bar-content-${currentContentId}`}
        openPanelId={0}
        closable={true}
        disableResize={true}
        enableCompact
        width={width}
      >
        <Sidebar.Panel panelId={0}>
          <div
            css={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              rowGap: theme.spacing.md,
              borderRadius: theme.legacyBorders.borderRadiusLg,
              border: `1px solid ${theme.colors.backgroundSecondary}`,
              padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
              backgroundColor: theme.colors.backgroundSecondary,
            }}
          >
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
                width: '100%',
              }}
            >
              <Typography.Text color="secondary">{title}</Typography.Text>
              <Button
                aria-label={closeLabel}
                icon={<CloseIcon />}
                componentId={`documentation-side-bar-close-${currentContentId}`}
                onClick={() => setCurrentContentId(undefined)}
              />
            </div>
            {content}
          </div>
        </Sidebar.Panel>
      </Sidebar.Content>
    </Sidebar>
  );
}
