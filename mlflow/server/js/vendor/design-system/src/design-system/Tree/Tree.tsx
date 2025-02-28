import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import type { TreeProps as AntDTreeProps } from 'antd';
import { Tree as AntDTree } from 'antd';
import chroma from 'chroma-js';
import { forwardRef } from 'react';

import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { ChevronDownIcon } from '../Icon';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export type { TreeDataNode } from 'antd';

export type NodeSize = 'default' | 'small' | 'x-small' | 'xx-small';
const hideLinesForSizes: Array<NodeSize> = ['x-small', 'xx-small'];

const sizeMap: Record<NodeSize, { nodeSize: number; indent: number }> = {
  default: { nodeSize: 32, indent: 28 },
  small: { nodeSize: 24, indent: 24 },
  'x-small': { nodeSize: 24, indent: 16 },
  'xx-small': { nodeSize: 24, indent: 8 },
};

export interface TreeProps extends HTMLDataAttributes, DangerouslySetAntdProps<AntDTreeProps> {
  /**
   * An array of `TreeNode`s representing the data in the tree
   */
  treeData: AntDTreeProps['treeData'];

  /**
   * An array of keys of nodes in the tree which should be in an expanded state when the tree is first rendered.
   */
  defaultExpandedKeys?: string[];

  /**
   * An array of keys of nodes in the tree which should be in a selected state when the tree is first rendered.
   * Use with `mode="selectable"`.
   */
  defaultSelectedKeys?: string[];

  /**
   * An array of keys of nodes in the tree which should be in a checked state when the tree is first rendered.
   * Use with `mode="checkable"`.
   */
  defaultCheckedKeys?: string[];

  /**
   * Visually and functionally disables the tree.
   */
  disabled?: boolean;

  /**
   * The UX mode of the tree. In `default` mode, nodes are for display only and are non-interactive.
   * In `selectable` mode, nodes can be selected and deselected. This is useful for something like a file browser.
   * In `checkable` mode, nodes are rendered with a checkbox. This is useful for something like a nested preference list.
   */
  mode?: 'default' | 'selectable' | 'multiselectable' | 'checkable';
  /**
   * The size of the button. Sizes below small continue to compact the tree horizontally but not vertically.
   */
  size?: NodeSize;

  /**
   * Whether to show connecting lines.
   */
  showLine?: boolean | { showLeafIcon: boolean };

  /**
   * (Controlled) Specifies the keys of the selected treeNodes, multiple selection needs to set multiple to true
   */
  selectedKeys?: AntDTreeProps['selectedKeys'];

  /**
   * (Controlled) Specifies the keys of the expanded treeNodes
   */
  expandedKeys?: AntDTreeProps['expandedKeys'];

  /**
   * Whether to automatically expand a parent treeNode
   */
  autoExpandParent?: AntDTreeProps['autoExpandParent'];

  /**
   * Callback function for when the user clicks a treeNode
   */
  onSelect?: AntDTreeProps['onSelect'];

  /**
   * Callback function for when a treeNode is expanded or collapsed
   */
  onExpand?: AntDTreeProps['onExpand'];

  /**
   * Callback function for when the onCheck event occurs
   */
  onCheck?: AntDTreeProps['onCheck'];
}

/**
 * These styles share some aspects with the styles in the main `Checkbox.tsx` component.
 * However, due to significant differences in the internal implementation and DOM structure of the Tree Checkbox and the
 * main Checkbox, we have forked the styles here.
 * Some notable differences are:
 * 1. Tree checkbox does not have a wrapper div
 * 2. Tree checkbox does not use a hidden input element
 * 3. Tree checkbox does not support the disabled state.
 * 4. Tree checkbox does not support keyboard focus
 */
function getTreeCheckboxEmotionStyles(clsPrefix: string, theme: Theme): CSSObject {
  const classRoot = `.${clsPrefix}`;
  const classInner = `.${clsPrefix}-inner`;
  const classIndeterminate = `.${clsPrefix}-indeterminate`;
  const classChecked = `.${clsPrefix}-checked`;
  const classDisabled = `.${clsPrefix}-disabled`;

  const styles: CSSObject = {
    [`${classRoot} > ${classInner}`]: {
      backgroundColor: theme.colors.actionDefaultBackgroundDefault,
      borderColor: theme.colors.actionDefaultBorderDefault,
    },

    // Hover
    [`${classRoot}:hover > ${classInner}`]: {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
    },

    // Mouse pressed
    [`${classRoot}:active > ${classInner}`]: {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      borderColor: theme.colors.actionDefaultBorderPress,
    },

    // Checked state
    [`${classChecked} > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
      borderColor: 'transparent',
    },

    // Checked hover
    [`${classChecked}:hover > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
      borderColor: 'transparent',
    },

    // Checked and mouse pressed
    [`${classChecked}:active > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundPress,
      borderColor: 'transparent',
    },

    // Indeterminate
    [`${classIndeterminate} > ${classInner}`]: {
      backgroundColor: theme.colors.primary,
      borderColor: theme.colors.primary,

      // The after pseudo-element is used for the check image itself
      '&:after': {
        backgroundColor: theme.colors.white,
        height: '3px',
        width: '8px',
        borderRadius: '4px',
      },
    },

    // Indeterminate hover
    [`${classIndeterminate}:hover > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
      borderColor: 'transparent',
    },

    // Indeterminate and mouse pressed
    [`${classIndeterminate}:active > ${classInner}`]: {
      backgroundColor: theme.colors.actionPrimaryBackgroundPress,
    },

    // Disabled
    [`${classDisabled} > ${classInner}, ${classDisabled}:hover > ${classInner}, ${classDisabled}:active > ${classInner}`]:
      {
        backgroundColor: theme.colors.actionDisabledBackground,
      },

    ...getAnimationCss(theme.options.enableAnimation),
  };

  return styles;
}

function getTreeEmotionStyles(clsPrefix: string, theme: Theme, size: NodeSize): SerializedStyles {
  const classNode = `.${clsPrefix}-tree-treenode`;
  const classNodeSelected = `.${clsPrefix}-tree-treenode-selected`;
  const classNodeActive = `.${clsPrefix}-tree-treenode-active`;
  const classNodeDisabled = `.${clsPrefix}-tree-treenode-disabled`;
  const classContent = `.${clsPrefix}-tree-node-content-wrapper`;
  const classContentTitle = `.${clsPrefix}-tree-title`;
  const classSelected = `.${clsPrefix}-tree-node-selected`;
  const classSwitcher = `.${clsPrefix}-tree-switcher`;
  const classSwitcherNoop = `.${clsPrefix}-tree-switcher-noop`;
  const classFocused = `.${clsPrefix}-tree-focused`;
  const classCheckbox = `.${clsPrefix}-tree-checkbox`;
  const classUnselectable = `.${clsPrefix}-tree-unselectable`;
  const classIndent = `.${clsPrefix}-tree-indent-unit`;
  const classTreeList = `.${clsPrefix}-tree-list`;
  const classScrollbar = `.${clsPrefix}-tree-list-scrollbar`;
  const classScrollbarThumb = `.${clsPrefix}-tree-list-scrollbar-thumb`;
  const classIcon = `.${clsPrefix}-tree-iconEle`;
  const classAntMotion = `.${clsPrefix}-tree-treenode-motion, .ant-motion-collapse-appear, .ant-motion-collapse-appear-active, .ant-motion-collapse`;

  const NODE_SIZE = sizeMap[size].nodeSize;
  const ICON_FONT_SIZE = 16;
  const BORDER_WIDTH = 4;

  const baselineAligned = {
    alignSelf: 'baseline',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const styles: CSSObject = {
    // Basic node
    [classNode]: {
      minHeight: NODE_SIZE,
      width: '100%',
      padding: 0,
      paddingLeft: BORDER_WIDTH,
      display: 'flex',
      alignItems: 'center',

      // Ant tree renders some hidden tree nodes (presumably for internal purposes). Setting these to width: 100% causes
      // overflow, so we need to reset here.
      '&[aria-hidden=true]': {
        width: 'auto',
      },

      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover,
      },

      '&:active': {
        backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      },
    },

    [`&${classUnselectable}`]: {
      // Remove hover and press styles if tree nodes are not selectable
      [classNode]: {
        '&:hover': {
          backgroundColor: 'transparent',
        },

        '&:active': {
          backgroundColor: 'transparent',
        },
      },

      [classContent]: {
        cursor: 'default',
      },

      // Unselectable nodes don't have any background, so the switcher looks better with rounded corners.
      [classSwitcher]: {
        borderRadius: theme.legacyBorders.borderRadiusMd,
      },
    },

    // The "active" node is the one that is currently focused via keyboard navigation. We give it the same visual
    // treatment as the mouse hover style.
    [classNodeActive]: {
      backgroundColor: theme.colors.actionTertiaryBackgroundHover,
    },

    // The "selected" node is one that has either been clicked on, or selected via pressing enter on the keyboard.
    [classNodeSelected]: {
      backgroundColor: theme.colors.actionDefaultBackgroundHover,
      borderLeft: `${BORDER_WIDTH}px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
      paddingLeft: 0,

      // When hovering over a selected node, we still want it to look selected
      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      },
    },

    [classSelected]: {
      background: 'none',
    },

    [classNodeDisabled]: {
      '&:hover': {
        backgroundColor: 'transparent',
      },
      '&:active': {
        backgroundColor: 'transparent',
      },
    },

    [classContent]: {
      lineHeight: `${NODE_SIZE}px`,
      // The content label is the interactive element, so we want it to fill the node to maximise the click area.
      width: '100%',
      display: 'flex',
      alignItems: 'center',

      '&:hover': {
        backgroundColor: 'transparent',
      },
      '&:active': {
        backgroundColor: 'transparent',
      },
    },
    [classContentTitle]: {
      lineHeight: theme.typography.lineHeightBase,
      padding: `${(NODE_SIZE - parseInt(theme.typography.lineHeightBase, 10)) / 2}px 0`,
      // The content inside 'classContent' is wrapped in the title class, which is the actual interactive element.
      width: '100%',
    },

    // TODO(FEINF-1595): Temporary style for now
    [`${classSwitcherNoop} + ${classContent}, ${classSwitcherNoop} + ${classCheckbox}`]: {
      marginLeft: NODE_SIZE + 4,
    },

    [classSwitcher]: {
      height: NODE_SIZE,
      width: NODE_SIZE,
      paddingTop: (NODE_SIZE - ICON_FONT_SIZE) / 2,
      marginRight: theme.spacing.xs,
      color: theme.colors.textSecondary,
      backgroundColor: 'transparent',

      // Keyboard navigation only allows moving between entire nodes, not between the switcher and label directly.
      // However, under mouse control, the two can still be clicked separately. We apply hover and press treatment
      // here to indicate to mouse users that the switcher is clickable.
      '&:hover': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover,
      },
      '&:active': {
        backgroundColor: theme.colors.actionTertiaryBackgroundPress,
      },
    },

    [classSwitcherNoop]: {
      display: 'none',

      '&:hover': {
        backgroundColor: 'transparent',
      },
      '&:active': {
        backgroundColor: 'transparent',
      },
    },

    [`&${classFocused}`]: {
      backgroundColor: 'transparent',
      outlineWidth: 2,
      outlineOffset: 1,
      outlineColor: theme.colors.actionDefaultBorderFocus,
      outlineStyle: 'solid',
    },

    [classCheckbox]: {
      marginTop: size === 'default' ? theme.spacing.sm : theme.spacing.xs,
      marginBottom: 0,
      marginRight: size === 'default' ? theme.spacing.sm : theme.spacing.xs,
      ...baselineAligned,
    },

    [classScrollbarThumb]: {
      background: chroma(theme.isDarkMode ? '#ffffff' : '#000000')
        .alpha(0.5)
        .hex(),
    },

    [`${classIcon}:has(*)`]: {
      ...baselineAligned,
      height: NODE_SIZE,
      color: theme.colors.textSecondary,
      marginRight: size === 'default' ? theme.spacing.sm : theme.spacing.xs,
    },

    // Needed to avoid flickering when has icon and expanding
    [classAntMotion]: {
      ...getAnimationCss(theme.options.enableAnimation),
      visibility: 'hidden',
    },

    // Vertical line
    [classIndent]: {
      width: sizeMap[size].indent,
    },

    [`${classIndent}:before`]: {
      height: '100%',
    },

    [classTreeList]: {
      [`&:hover ${classScrollbar}`]: { display: 'block !important' },
      [`&:active ${classScrollbar}`]: { display: 'block !important' },
    },

    ...getTreeCheckboxEmotionStyles(`${clsPrefix}-tree-checkbox`, theme),

    ...getAnimationCss(theme.options.enableAnimation),
  };

  const importantStyles = importantify(styles);

  return css(importantStyles);
}

const SHOW_LINE_DEFAULT = { showLeafIcon: false };

// @ts-expect-error: Tree doesn't expose a proper type
export const Tree = forwardRef<AntDTree, TreeProps>(function Tree(
  {
    treeData,
    defaultExpandedKeys,
    defaultSelectedKeys,
    defaultCheckedKeys,
    disabled = false,
    mode = 'default',
    size = 'default',
    showLine,
    dangerouslySetAntdProps,
    ...props
  }: TreeProps,
  ref,
): JSX.Element {
  const { theme, classNamePrefix } = useDesignSystemTheme();

  let calculatedShowLine = showLine ?? false;
  if (hideLinesForSizes.includes(size)) {
    calculatedShowLine = false;
  } else {
    calculatedShowLine = showLine ?? SHOW_LINE_DEFAULT;
  }

  return (
    <DesignSystemAntDConfigProvider>
      <AntDTree
        {...addDebugOutlineIfEnabled()}
        treeData={treeData}
        defaultExpandedKeys={defaultExpandedKeys}
        defaultSelectedKeys={defaultSelectedKeys}
        defaultCheckedKeys={defaultCheckedKeys}
        disabled={disabled}
        css={getTreeEmotionStyles(classNamePrefix, theme, size)}
        switcherIcon={<ChevronDownIcon css={{ fontSize: '16px !important' }} />}
        tabIndex={0}
        selectable={mode === 'selectable' || mode === 'multiselectable'}
        checkable={mode === 'checkable'}
        multiple={mode === 'multiselectable'}
        // With the library flag, defaults to showLine = true. The status quo default is showLine = false.
        showLine={calculatedShowLine}
        {...dangerouslySetAntdProps}
        {...props}
        ref={ref}
      />
    </DesignSystemAntDConfigProvider>
  );
});
