import React from 'react';
import { Collapse, Icon } from 'antd';
import { SectionErrorBoundary } from './error-boundaries/SectionErrorBoundary';
import PropTypes from 'prop-types';
import { css } from 'emotion';

export function CollapsibleSection(props) {
  const { title, forceOpen, showServerError } = props;
  // We need to spread `activeKey` into <Collapse/> as an optional prop because its enumerability
  // affects rendering, i.e. passing `activeKey={undefined}` is different from not passing activeKey
  const activeKeyProp = forceOpen && { activeKey: ['1'] };
  return (
    <Collapse
      className={`collapsible-section ${classNames.wrapper}`}
      bordered={false}
      {...activeKeyProp}
      defaultActiveKey={['1']}
      expandIcon={({ isActive }) => <Icon type='caret-right' rotate={isActive ? 90 : 0} />}
    >
      <Collapse.Panel className='collapsible-panel' header={title} key='1'>
        <SectionErrorBoundary showServerError={showServerError}>
          {props.children}
        </SectionErrorBoundary>
      </Collapse.Panel>
    </Collapse>
  );
}

const classNames = {
  wrapper: css({
    '.collapsible-panel': {
      position: 'relative',
    },
  }),
};

CollapsibleSection.propTypes = {
  title: PropTypes.oneOfType([PropTypes.string, PropTypes.object]).isRequired,
  forceOpen: PropTypes.bool,
  children: PropTypes.node.isRequired,
  showServerError: PropTypes.bool,
  // when true, if an error is triggered, SectionErrorBoundary will show the server error
};

CollapsibleSection.defaultProps = {
  forceOpen: false,
};
