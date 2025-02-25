import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import type { ModalProps as AntDModalProps } from 'antd';
import { Modal as AntDModal } from 'antd';
import React, { createContext, useContext, useMemo } from 'react';

import type { Theme } from '../../theme';
import type { ButtonProps } from '../Button';
import { Button } from '../Button';
import {
  augmentWithDataComponentProps,
  useDesignSystemEventComponentCallbacks,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import {
  DesignSystemEventSuppressInteractionProviderContext,
  DesignSystemEventSuppressInteractionTrueContextValue,
} from '../DesignSystemEventProvider/DesignSystemEventSuppressInteractionProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon, DangerIcon } from '../Icon';
import type { AnalyticsEventPropsWithStartInteraction, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { getDarkModePortalStyles, getShadowScrollStyles, useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled, addDebugOutlineStylesIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
import { useNotifyOnFirstView } from '../utils/useNotifyOnFirstView';

export interface ModalProps
  extends HTMLDataAttributes,
    DangerouslySetAntdProps<AntDModalProps>,
    AnalyticsEventPropsWithStartInteraction<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
  /** Whether or not the modal is currently open. Use together with onOk and onCancel to control the modal state. */
  visible?: AntDModalProps['visible'];

  /** Function called when the primary button is clicked */
  onOk?: AntDModalProps['onOk'];

  /** Function called when the secondary button is clicked */
  onCancel?: AntDModalProps['onCancel'];

  /** Title displayed at the top of the modal */
  title?: AntDModalProps['title'];

  /** Contents displayed in the body of the modal */
  children?: React.ReactNode;

  /** A custom JSX element to render in place of the default footer. If `footer` is not provided or is set to `undefined`, the default footer will be rendered. `footer` must be explicitly set to `null` to hide the footer. */
  footer?: AntDModalProps['footer'];

  /** Sets the horizontal size according to the size presets */
  size?: 'normal' | 'wide';

  /** Defines the modal height either by both its content and maximum height (dynamic) or by its maximum height only (maxed_out) */
  verticalSizing?: 'dynamic' | 'maxed_out';

  /** When set to true, the confirm button will show a loading indicator */
  confirmLoading?: AntDModalProps['confirmLoading'];

  /** Function called after the modal has finished closing */
  afterClose?: AntDModalProps['afterClose'];

  /** Text of the primary button */
  okText?: AntDModalProps['okText'];

  /** Text of the secondary button */
  cancelText?: AntDModalProps['cancelText'];

  /** Set to true to force the modal to render */
  forceRender?: AntDModalProps['forceRender'];

  /** When true, the modal content will be destroyed when closed, rather than just visually hidden. */
  destroyOnClose?: AntDModalProps['destroyOnClose'];

  /** A CSS class name to apply to the modal wrapper */
  wrapClassName?: AntDModalProps['wrapClassName'];

  /** A CSS class name to apply to the modal */
  className?: AntDModalProps['className'];

  /** A function that returns the element into which the modal will be rendered */
  getContainer?: AntDModalProps['getContainer'];

  /**
   * Allows setting the CSS z-index of the modal.
   * Can be used to work around ordering issues when the modal is not overlaying other elements correctly
   */
  zIndex?: AntDModalProps['zIndex'];

  /** The OK button props */
  okButtonProps?: Omit<ButtonProps, 'componentId' | 'analyticsEvents'>;

  /** The cancel button props */
  cancelButtonProps?: Omit<ButtonProps, 'componentId' | 'analyticsEvents'>;

  /** Custom modal content render function. While custom rendering is discouraged in general, this prop can be used to wrap the modal contents in a React context. */
  modalRender?: (node: React.ReactNode) => React.ReactNode;

  /** Specify which button to autofocus, only works with default footer */
  autoFocusButton?: 'ok' | 'cancel';

  /** Specify whether to truncate the Modal title when it is too long */
  truncateTitle?: boolean;
}

export interface ModalContextProps {
  isInsideModal: boolean;
}

const ModalContext = createContext<ModalContextProps>({
  isInsideModal: true,
});

export const useModalContext = () => useContext(ModalContext);

const SIZE_PRESETS = {
  normal: 640,
  wide: 880,
};

const getModalEmotionStyles = (args: {
  theme: Theme;
  clsPrefix: string;
  hasFooter: boolean;
  maxedOutHeight: boolean;
  useNewShadows: boolean;
}): SerializedStyles => {
  const { theme, clsPrefix, hasFooter = true, maxedOutHeight, useNewShadows } = args;

  const classNameClose = `.${clsPrefix}-modal-close`;
  const classNameCloseX = `.${clsPrefix}-modal-close-x`;
  const classNameTitle = `.${clsPrefix}-modal-title`;
  const classNameContent = `.${clsPrefix}-modal-content`;
  const classNameBody = `.${clsPrefix}-modal-body`;
  const classNameHeader = `.${clsPrefix}-modal-header`;
  const classNameFooter = `.${clsPrefix}-modal-footer`;
  const classNameButton = `.${clsPrefix}-btn`;
  const classNameDropdownTrigger = `.${clsPrefix}-dropdown-button`;

  const MODAL_PADDING = theme.spacing.lg;
  const BUTTON_SIZE = theme.general.heightSm;
  // Needed for moving some of the padding from the header and footer to the content to avoid a scrollbar from appearing
  // when the content has some interior components that reach the limits of the content div
  // 8px is an arbitrary value, it still leaves enough padding for the header and footer too to avoid the same problem
  // from occurring there too
  const CONTENT_BUFFER = 8;

  const modalMaxHeight = '90vh';
  const headerHeight = 64;
  const footerHeight = hasFooter ? 52 : 0;
  const bodyMaxHeight = `calc(${modalMaxHeight} - ${headerHeight}px - ${footerHeight}px - ${MODAL_PADDING}px)`;

  return css({
    '&&': {
      ...addDebugOutlineStylesIfEnabled(theme),
    },

    [classNameHeader]: {
      background: 'transparent',
      paddingTop: theme.spacing.md,
      paddingLeft: theme.spacing.lg,
      paddingRight: theme.spacing.md,
      paddingBottom: theme.spacing.md,
    },
    [classNameFooter]: {
      height: footerHeight,
      paddingTop: theme.spacing.lg - CONTENT_BUFFER,
      paddingLeft: MODAL_PADDING,
      paddingRight: MODAL_PADDING,
      marginTop: 'auto',

      [`${classNameButton} + ${classNameButton}`]: {
        marginLeft: theme.spacing.sm,
      },
      // Needed to override AntD style for the SplitButton's dropdown button back to its original value
      [`${classNameDropdownTrigger} > ${classNameButton}:nth-of-type(2)`]: {
        marginLeft: -1,
      },
    },
    [classNameCloseX]: {
      fontSize: theme.general.iconSize,
      height: BUTTON_SIZE,
      width: BUTTON_SIZE,
      lineHeight: 'normal',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: theme.colors.textSecondary,
    },
    [classNameClose]: {
      height: BUTTON_SIZE,
      width: BUTTON_SIZE,
      // Note: Ant has the close button absolutely positioned, rather than in a flex container with the title.
      // This magic number is eyeballed to get the close X to align with the title text.
      margin: '16px 16px 0 0',
      borderRadius: theme.legacyBorders.borderRadiusMd,

      backgroundColor: theme.colors.actionDefaultBackgroundDefault,
      borderColor: theme.colors.actionDefaultBackgroundDefault,
      color: theme.colors.actionDefaultTextDefault,

      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        borderColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionDefaultTextHover,
      },

      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        borderColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress,
      },

      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '1px',
        outlineColor: theme.colors.actionDefaultBorderFocus,
      },
    },
    [classNameTitle]: {
      fontSize: theme.typography.fontSizeXl,
      lineHeight: theme.typography.lineHeightXl,
      fontWeight: theme.typography.typographyBoldFontWeight,
      paddingRight: MODAL_PADDING,
      minHeight: headerHeight / 2,
      display: 'flex',
      alignItems: 'center',
      overflowWrap: 'anywhere',
    },
    [classNameContent]: {
      backgroundColor: theme.colors.backgroundPrimary,
      maxHeight: modalMaxHeight,
      height: maxedOutHeight ? modalMaxHeight : '',
      overflow: 'hidden',
      paddingBottom: MODAL_PADDING,
      display: 'flex',
      flexDirection: 'column',
      boxShadow: useNewShadows ? theme.shadows.xl : theme.general.shadowHigh,
      ...getDarkModePortalStyles(theme, useNewShadows),
    },
    [classNameBody]: {
      overflowY: 'auto',
      maxHeight: bodyMaxHeight,
      paddingLeft: MODAL_PADDING,
      paddingRight: MODAL_PADDING,
      paddingTop: CONTENT_BUFFER,
      paddingBottom: CONTENT_BUFFER,

      ...getShadowScrollStyles(theme),
    },

    ...getAnimationCss(theme.options.enableAnimation),
  });
};

function closeButtonComponentId(componentId: string | undefined): string {
  return componentId ? `${componentId}.footer.cancel` : 'codegen_design-system_src_design-system_modal_modal.tsx_260';
}

/**
 * Render default footer with our buttons. Copied from AntD.
 */
function DefaultFooter({
  componentId,
  onOk,
  onCancel,
  confirmLoading,
  okText,
  cancelText,
  okButtonProps,
  cancelButtonProps,
  autoFocusButton,
  shouldStartInteraction,
}: Pick<
  ModalProps,
  | 'componentId'
  | 'onOk'
  | 'onCancel'
  | 'confirmLoading'
  | 'okText'
  | 'cancelText'
  | 'okButtonProps'
  | 'cancelButtonProps'
  | 'autoFocusButton'
  | 'shouldStartInteraction'
>) {
  const handleCancel = (e: React.MouseEvent<HTMLButtonElement>) => {
    onCancel?.(e);
  };

  const handleOk = (e: React.MouseEvent<HTMLButtonElement>) => {
    onOk?.(e);
  };

  return (
    <>
      {cancelText && (
        <Button
          componentId={closeButtonComponentId(componentId)}
          onClick={handleCancel}
          autoFocus={autoFocusButton === 'cancel'}
          dangerouslyUseFocusPseudoClass
          shouldStartInteraction={shouldStartInteraction}
          {...cancelButtonProps}
        >
          {cancelText}
        </Button>
      )}
      {okText && (
        <Button
          componentId={
            componentId ? `${componentId}.footer.ok` : 'codegen_design-system_src_design-system_modal_modal.tsx_271'
          }
          loading={confirmLoading}
          onClick={handleOk}
          type="primary"
          autoFocus={autoFocusButton === 'ok'}
          dangerouslyUseFocusPseudoClass
          shouldStartInteraction={shouldStartInteraction}
          {...okButtonProps}
        >
          {okText}
        </Button>
      )}
    </>
  );
}

export function Modal(props: ModalProps): JSX.Element {
  return (
    <DesignSystemEventSuppressInteractionProviderContext.Provider
      value={DesignSystemEventSuppressInteractionTrueContextValue}
    >
      <ModalInternal {...props} />
    </DesignSystemEventSuppressInteractionProviderContext.Provider>
  );
}

function ModalInternal({
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
  okButtonProps,
  cancelButtonProps,
  dangerouslySetAntdProps,
  children,
  title,
  footer,
  size = 'normal',
  verticalSizing = 'dynamic',
  autoFocusButton,
  truncateTitle,
  shouldStartInteraction,
  ...props
}: ModalProps): JSX.Element {
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Modal,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    shouldStartInteraction,
  });
  const { elementRef } = useNotifyOnFirstView<HTMLElement>({ onView: eventContext.onView });

  // Need to simulate the close button being closed if the user clicks outside of the modal or clicks on the dismiss button
  // This should only be applied to the modal prop and not to the footer component
  const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: closeButtonComponentId(componentId),
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    shouldStartInteraction,
  });

  const onCancelWrapper = (e: React.MouseEvent<HTMLElement, MouseEvent>) => {
    closeButtonEventContext.onClick(e);
    props.onCancel?.(e);
  };

  // add data-component-* props to make modal discoverable by go/component-finder
  const augmentedChildren = safex('databricks.fe.observability.enableModalDataComponentProps', false)
    ? augmentWithDataComponentProps(children, eventContext.dataComponentProps)
    : children;

  return (
    <DesignSystemAntDConfigProvider>
      <AntDModal
        {...addDebugOutlineIfEnabled()}
        css={getModalEmotionStyles({
          theme,
          clsPrefix: classNamePrefix,
          hasFooter: footer !== null,
          maxedOutHeight: verticalSizing === 'maxed_out',
          useNewShadows,
        })}
        title={
          <RestoreAntDDefaultClsPrefix>
            {truncateTitle ? (
              <div
                css={{
                  textOverflow: 'ellipsis',
                  marginRight: theme.spacing.md,
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                }}
                title={typeof title === 'string' ? title : undefined}
              >
                {title}
              </div>
            ) : (
              title
            )}
          </RestoreAntDDefaultClsPrefix>
        }
        footer={
          footer === null ? null : (
            <RestoreAntDDefaultClsPrefix>
              {footer === undefined ? (
                <DefaultFooter
                  componentId={componentId}
                  onOk={props.onOk}
                  onCancel={props.onCancel}
                  confirmLoading={props.confirmLoading}
                  okText={props.okText}
                  cancelText={props.cancelText}
                  okButtonProps={okButtonProps}
                  cancelButtonProps={cancelButtonProps}
                  autoFocusButton={autoFocusButton}
                  shouldStartInteraction={shouldStartInteraction}
                />
              ) : (
                footer
              )}
            </RestoreAntDDefaultClsPrefix>
          )
        }
        width={size ? SIZE_PRESETS[size] : undefined}
        closeIcon={<CloseIcon ref={elementRef} />}
        centered
        zIndex={theme.options.zIndexBase}
        maskStyle={{
          backgroundColor: theme.colors.overlayOverlay,
        }}
        {...props}
        onCancel={onCancelWrapper}
        {...dangerouslySetAntdProps}
      >
        <RestoreAntDDefaultClsPrefix>
          <ModalContext.Provider value={{ isInsideModal: true }}>{augmentedChildren}</ModalContext.Provider>
        </RestoreAntDDefaultClsPrefix>
      </AntDModal>
    </DesignSystemAntDConfigProvider>
  );
}

export function DangerModal(props: Omit<ModalProps, 'footer'>): JSX.Element {
  const { theme } = useDesignSystemTheme();

  const { title, onCancel, onOk, cancelText, okText, okButtonProps, cancelButtonProps, ...restProps } = props;
  const iconSize = 18;
  const iconFontSize = 18;

  const titleComp = (
    <div css={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <DangerIcon
        css={{
          color: theme.colors.textValidationDanger,
          left: 2,
          height: iconSize,
          width: iconSize,
          fontSize: iconFontSize,
        }}
      />
      <div css={{ paddingLeft: 6 }}>{title}</div>
    </div>
  );

  return (
    <Modal
      shouldStartInteraction={props.shouldStartInteraction}
      title={titleComp}
      footer={[
        <Button
          componentId={
            props.componentId
              ? `${props.componentId}.danger.footer.cancel`
              : 'codegen_design-system_src_design-system_modal_modal.tsx_386'
          }
          key="cancel"
          onClick={onCancel}
          shouldStartInteraction={props.shouldStartInteraction}
          {...cancelButtonProps}
        >
          {cancelText || 'Cancel'}
        </Button>,

        <Button
          componentId={
            props.componentId
              ? `${props.componentId}.danger.footer.ok`
              : 'codegen_design-system_src_design-system_modal_modal.tsx_395'
          }
          key="discard"
          type="primary"
          danger
          onClick={onOk}
          loading={props.confirmLoading}
          shouldStartInteraction={props.shouldStartInteraction}
          {...okButtonProps}
        >
          {okText || 'Delete'}
        </Button>,
      ]}
      onOk={onOk}
      onCancel={onCancel}
      {...restProps}
    />
  );
}
