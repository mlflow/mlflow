import React from 'react';
import PropTypes from 'prop-types';
import { Button as AntdButton } from './antd/Button';
import { css, cx } from 'emotion';
import { buttonPrimaryBackground, buttonPrimaryBackgroundHover } from '../colors';

/**
 * Render a wrapper component around ANTD button. Styled according to
 * Unified ML design.
 * @param props onClick: function to run on button click.
 * @param props type: one of "primary" or "default".
 * @param props disabled: whether button click is disabled.
 * @param props dataTestId: data-test-id attribute, used for integration testing.
 */
export class Button extends React.Component {
  static propTypes = {
    children: PropTypes.node,
    onClick: PropTypes.func,
    size: PropTypes.oneOf(['small', 'middle', 'large']),
    type: PropTypes.string,
    disabled: PropTypes.bool,
    dataTestId: PropTypes.string,
    className: PropTypes.string,
  };

  render() {
    const {
      onClick,
      children,
      size = 'large',
      type,
      disabled,
      dataTestId,
      ...restProps
    } = this.props;
    return (
      <div className={cx(btnTextClassName, this.props.className)} data-test-id={dataTestId}>
        <AntdButton onClick={onClick} type={type} disabled={disabled} size={size} {...restProps}>
          {children}
        </AntdButton>
      </div>
    );
  }
}

const btnTextClassName = css({
  // override bootstrap-style webpack config
  '--text-selected-background-color': 'auto',
  '> .ant-btn-primary:not([disabled])': {
    background: `var(--button-primary-background-color, ${buttonPrimaryBackground})`,
    borderColor: `var(--button-primary-background-color, ${buttonPrimaryBackground})`,
    '&:hover': {
      background: `var(--button-primary-background-hover-color, ${buttonPrimaryBackgroundHover})`,
      borderColor: `var(--button-primary-background-hover-color, ${buttonPrimaryBackgroundHover})`,
    },
  },
  '& .ant-btn-lg': {
    fontSize: '14px',
  },
});
