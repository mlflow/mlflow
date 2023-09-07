/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Collapse } from 'antd';
import { SectionErrorBoundary } from './error-boundaries/SectionErrorBoundary';
import { ChevronRightIcon } from '@databricks/design-system';

type OwnProps = {
  title: string | any;
  forceOpen?: boolean;
  children: React.ReactNode;
  showServerError?: boolean;
  defaultCollapsed?: boolean;
  onChange?: (...args: any[]) => any;
};

// @ts-expect-error TS(2565): Property 'defaultProps' is used before being assig... Remove this comment to see the full error message
type Props = OwnProps & typeof CollapsibleSection.defaultProps;

export function CollapsibleSection(props: Props) {
  const { title, forceOpen, showServerError, defaultCollapsed, onChange } = props;
  // We need to spread `activeKey` into <Collapse/> as an optional prop because its enumerability
  // affects rendering, i.e. passing `activeKey={undefined}` is different from not passing activeKey
  const activeKeyProp = forceOpen && { activeKey: ['1'] };
  const defaultActiveKey = defaultCollapsed ? null : ['1'];
  return (
    <Collapse
      className='collapsible-section'
      // @ts-expect-error TS(2322): Type '{ '.collapsible-panel': { position: string; ... Remove this comment to see the full error message
      css={classNames.wrapper}
      bordered={false}
      {...activeKeyProp}
      // @ts-expect-error TS(2322): Type 'string[] | null' is not assignable to type '... Remove this comment to see the full error message
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
    '& > .collapsible-panel > [role="button"]:focus': {
      outline: 'revert',
    },
  },
};

CollapsibleSection.defaultProps = {
  forceOpen: false,
};
