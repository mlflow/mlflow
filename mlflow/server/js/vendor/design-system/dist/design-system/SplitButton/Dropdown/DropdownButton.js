import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import EllipsisOutlined from '@ant-design/icons';
import { Button as AntDButton, Dropdown as AntDDropdown } from 'antd';
import classNames from 'classnames';
import * as React from 'react';
import { Button } from '../../Button';
import { DropdownMenu } from '../../DropdownMenu';
import { useDesignSystemContext } from '../../Hooks/useDesignSystemContext';
import { ChevronDownIcon } from '../../Icon';
import { useDesignSystemSafexFlags } from '../../utils';
const ButtonGroup = AntDButton.Group;
export const DropdownButton = (props) => {
    const { getPopupContainer: getContextPopupContainer, getPrefixCls } = useDesignSystemContext();
    const { useNewBorderRadii } = useDesignSystemSafexFlags();
    const { type, danger, disabled, loading, onClick, htmlType, children, className, overlay, trigger, align, open, onOpenChange, placement, getPopupContainer, href, icon = _jsx(EllipsisOutlined, {}), title, buttonsRender = (buttons) => buttons, mouseEnterDelay, mouseLeaveDelay, overlayClassName, overlayStyle, destroyPopupOnHide, menuButtonLabel = 'Open dropdown', menu, leftButtonIcon, dropdownMenuRootProps, 'aria-label': ariaLabel, componentId, analyticsEvents, ...restProps } = props;
    const prefixCls = getPrefixCls('dropdown-button');
    const dropdownProps = {
        align,
        overlay,
        disabled,
        trigger: disabled ? [] : trigger,
        onOpenChange,
        getPopupContainer: getPopupContainer || getContextPopupContainer,
        mouseEnterDelay,
        mouseLeaveDelay,
        overlayClassName,
        overlayStyle,
        destroyPopupOnHide,
    };
    if ('open' in props) {
        dropdownProps.open = open;
    }
    if ('placement' in props) {
        dropdownProps.placement = placement;
    }
    else {
        dropdownProps.placement = 'bottomRight';
    }
    const leftButton = (_jsxs(Button, { componentId: componentId
            ? `${componentId}.primary_button`
            : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_148', type: type, danger: danger, disabled: disabled, loading: loading, onClick: onClick, htmlType: htmlType, href: href, title: title, icon: children && leftButtonIcon ? leftButtonIcon : undefined, "aria-label": ariaLabel, css: useNewBorderRadii ?? {
            borderTopRightRadius: '0 !important',
            borderBottomRightRadius: '0 !important',
        }, children: [leftButtonIcon && !children ? leftButtonIcon : undefined, children] }));
    const rightButton = (_jsx(Button, { componentId: componentId
            ? `${componentId}.dropdown_button`
            : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_166', type: type, danger: danger, disabled: disabled, "aria-label": menuButtonLabel, css: useNewBorderRadii ?? {
            borderTopLeftRadius: '0 !important',
            borderBottomLeftRadius: '0 !important',
        }, children: icon ? icon : _jsx(ChevronDownIcon, {}) }));
    const [leftButtonToRender, rightButtonToRender] = buttonsRender([leftButton, rightButton]);
    return (_jsxs(ButtonGroup, { ...restProps, className: classNames(prefixCls, className), children: [leftButtonToRender, overlay !== undefined ? (_jsx(AntDDropdown, { ...dropdownProps, overlay: overlay, children: rightButtonToRender })) : (_jsxs(DropdownMenu.Root, { ...dropdownMenuRootProps, itemHtmlType: htmlType === 'submit' ? 'submit' : undefined, children: [_jsx(DropdownMenu.Trigger, { disabled: disabled, asChild: true, children: rightButtonToRender }), menu && React.cloneElement(menu, { align: menu.props.align || 'end' })] }))] }));
};
//# sourceMappingURL=DropdownButton.js.map