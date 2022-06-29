import React from 'react';
import PropTypes from 'prop-types';
import { Button } from 'antd';
import { Dropdown } from '@databricks/design-system';
import expandIcon from '../static/expand-more.svg';
import { getUUID } from '../utils/ActionUtils';

export const StyledDropdown = ({ id, className, title, triggers, dropdownOptions, buttonSize }) => {
  return (
    <div css={classNames.wrapper}>
      <Dropdown
        id={id}
        className={className}
        title={title}
        trigger={triggers}
        overlay={dropdownOptions}
      >
        <Button className='StyledDropdown-button' size={buttonSize}>
          <div className='StyledDropdown-button-content'>
            <span>{title}</span>{' '}
            <img className='StyledDropdown-chevron' src={expandIcon} alt='Expand' />
          </div>
        </Button>
      </Dropdown>
    </div>
  );
};

const classNames = {
  wrapper: {
    display: 'inline-block',
    '.StyledDropdown-button': {
      padding: 0,
      color: '#1D2528',
    },
    '.StyledDropdown-button-content': {
      paddingLeft: '16px',
      paddingRight: '16px',
      display: 'flex',
      alignItems: 'center',
    },
  },
};

StyledDropdown.propTypes = {
  dropdownOptions: PropTypes.node.isRequired,
  title: PropTypes.string.isRequired,
  buttonSize: PropTypes.string,
  triggers: PropTypes.array,
  className: PropTypes.string,
  id: PropTypes.string,
  restProps: PropTypes.object,
};

StyledDropdown.defaultProps = {
  triggers: ['click'],
  className: 'StyledDropdown',
  id: 'StyledDropdown' + getUUID(),
  buttonSize: 'default',
};
