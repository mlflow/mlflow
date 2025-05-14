import type { TreeProps as AntDTreeProps } from 'antd';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export type { TreeDataNode } from 'antd';
export type NodeSize = 'default' | 'small' | 'x-small' | 'xx-small';
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
    showLine?: boolean | {
        showLeafIcon: boolean;
    };
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
export declare const Tree: import("react").ForwardRefExoticComponent<TreeProps & import("react").RefAttributes<AntDTree>>;
//# sourceMappingURL=Tree.d.ts.map