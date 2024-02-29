/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Button, Dropdown } from '@databricks/design-system';
import expandIcon from '../static/expand-more.svg';
import { getUUID } from '../utils/ActionUtils';

type OwnProps = {
  dropdownOptions: React.ReactNode;
  title: string;
  buttonSize?: string;
  triggers?: any[];
  className?: string;
  id?: string;
  restProps?: any;
};

// @ts-expect-error TS(2456): Type alias 'Props' circularly references itself.
type Props = OwnProps & typeof StyledDropdown.defaultProps;

// @ts-expect-error TS(7022): 'StyledDropdown' implicitly has type 'any' because... Remove this comment to see the full error message
export const StyledDropdown = ({ id, className, title, triggers, dropdownOptions, buttonSize }: Props) => {
  return (
    <div css={classNames.wrapper}>
      <Dropdown
        // @ts-expect-error TS(2322): Type '{ children: Element; id: any; className: any... Remove this comment to see the full error message
        id={id}
        className={className}
        title={title}
        trigger={triggers}
        overlay={dropdownOptions}
      >
        <Button
          componentId="codegen_mlflow_app_src_common_components_styleddropdown.tsx_38"
          className="StyledDropdown-button"
          size={buttonSize}
          css={classNames.button}
        >
          <span>{title}</span> <img className="StyledDropdown-chevron" src={expandIcon} alt="Expand" />
        </Button>
      </Dropdown>
    </div>
  );
};

const classNames = {
  button: (theme: any) => ({
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

StyledDropdown.defaultProps = {
  triggers: ['click'],
  className: 'StyledDropdown',
  id: 'StyledDropdown' + getUUID(),
  buttonSize: 'default',
};
