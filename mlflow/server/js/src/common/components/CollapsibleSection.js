import React from 'react';
import { Collapse, Icon } from 'antd';
import PropTypes from 'prop-types';

export function CollapsibleSection(props) {
  const { title, forceOpen } = props;
  // We need to spread `activeKey` into <Collapse/> as an optional prop because its enumerability
  // affects rendering, i.e. passing `activeKey={undefined}` is different from not passing activeKey
  const activeKeyProp = forceOpen && { activeKey: ['1'] };
  return (
    <Collapse
      className='collapsible-section'
      bordered={false}
      {...activeKeyProp}
      defaultActiveKey={['1']}
      expandIcon={({ isActive }) => <Icon type='caret-right' rotate={isActive ? 90 : 0} />}
    >
      <Collapse.Panel className='collapsible-panel' header={title} key='1'>
        {props.children}
      </Collapse.Panel>
    </Collapse>
  );
}

CollapsibleSection.propTypes = {
  title: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.object,
  ]).isRequired,
  forceOpen: PropTypes.bool,
  children: PropTypes.object,
};

CollapsibleSection.defaultProps = {
  forceOpen: false,
};
