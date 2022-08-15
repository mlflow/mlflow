import React from 'react';
import { Collapse } from 'antd';
import { SectionErrorBoundary } from './error-boundaries/SectionErrorBoundary';
import PropTypes from 'prop-types';
import { ChevronRightIcon } from '@databricks/design-system';

export function CollapsibleSection(props) {
  const { title, forceOpen, showServerError, defaultCollapsed, onChange } = props;
  // We need to spread `activeKey` into <Collapse/> as an optional prop because its enumerability
  // affects rendering, i.e. passing `activeKey={undefined}` is different from not passing activeKey
  const activeKeyProp = forceOpen && { activeKey: ['1'] };
  const defaultActiveKey = defaultCollapsed ? null : ['1'];
  return (
    <Collapse
      className='collapsible-section'
      css={classNames.wrapper}
      bordered={false}
      {...activeKeyProp}
      defaultActiveKey={defaultActiveKey}
      expandIcon={({ isActive }) => (
        // Font-size !important because antd's css clobbers any sort of svg size here.
        <ChevronRightIcon css={{ fontSize: '20px!important' }} rotate={isActive ? 90 : 0} />
      )}
      onChange={onChange}
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
  wrapper: {
    '.collapsible-panel': {
      position: 'relative',
    },
  },
};

CollapsibleSection.propTypes = {
  title: PropTypes.oneOfType([PropTypes.string, PropTypes.object]).isRequired,
  forceOpen: PropTypes.bool,
  children: PropTypes.node.isRequired,
  showServerError: PropTypes.bool,
  defaultCollapsed: PropTypes.bool,
  // when true, if an error is triggered, SectionErrorBoundary will show the server error
  onChange: PropTypes.func,
};

CollapsibleSection.defaultProps = {
  forceOpen: false,
};
