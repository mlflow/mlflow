import React from 'react';
import PropTypes from 'prop-types';
import {
  Breadcrumb,
  Button,
  Spacer,
  Dropdown,
  Menu,
  Header,
  OverflowIcon,
} from '@databricks/design-system';
import { PreviewIcon } from './PreviewIcon';

// Note: this button has a different size from normal AntD buttons.
export { Button as HeaderButton };

export function OverflowMenu({ menu }) {
  const overflowMenu = (
    <Menu>
      {menu.map(({ id, itemName, onClick, href, ...otherProps }) => (
        <Menu.Item key={id} onClick={onClick} href={href} data-test-id={id} {...otherProps}>
          {itemName}
        </Menu.Item>
      ))}
    </Menu>
  );

  return menu.length > 0 ? (
    <Dropdown overlay={overflowMenu} trigger={['click']} placement='bottomLeft' arrow>
      <Button icon={<OverflowIcon />} data-test-id='overflow-menu-trigger' />
    </Dropdown>
  ) : null;
}

OverflowMenu.propTypes = {
  menu: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      itemName: PropTypes.node.isRequired,
      onClick: PropTypes.func,
      href: PropTypes.string,
    }),
  ),
};

/**
 * A page header that includes:
 *   - title,
 *   - optional breadcrumb content,
 *   - optional preview mark,
 *   - optional feedback link, and
 *   - optional info popover, safe to have link inside.
 */
export function PageHeader({
  title, // required
  breadcrumbs = [],
  preview,
  children,
}) {
  return (
    <>
      <Header
        breadcrumbs={
          breadcrumbs.length > 0 && (
            <Breadcrumb includeTrailingCaret>
              {breadcrumbs.map((b, i) => (
                <Breadcrumb.Item key={i}>{b}</Breadcrumb.Item>
              ))}
            </Breadcrumb>
          )
        }
        buttons={children}
        title={title}
        // prettier-ignore
        titleAddOns={
          <>
            {preview && <PreviewIcon />}
          </>
        }
      />
      <Spacer />
    </>
  );
}

PageHeader.propTypes = {
  title: PropTypes.node.isRequired,
  breadcrumbs: PropTypes.arrayOf(PropTypes.node),
  preview: PropTypes.bool,
  children: PropTypes.node,
};
