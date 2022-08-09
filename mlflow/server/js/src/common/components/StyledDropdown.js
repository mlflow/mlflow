import React from 'react';
import PropTypes from 'prop-types';
import { Button, Dropdown } from '@databricks/design-system';
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
        <Button className='StyledDropdown-button' size={buttonSize} css={classNames.button}>
          <span>{title}</span>{' '}
          <img className='StyledDropdown-chevron' src={expandIcon} alt='Expand' />
        </Button>
      </Dropdown>
    </div>
  );
};

const classNames = {
  button: (theme) => ({
    fontSize: theme.typography.fontSizeBase,
  }),
  wrapper: {
    display: 'inline-block',
    '.StyledDropdown-button': {
      padding: 0,
      color: '#1D2528',
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
