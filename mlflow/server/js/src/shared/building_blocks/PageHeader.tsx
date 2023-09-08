/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import {
  Breadcrumb,
  Button,
  Spacer,
  Dropdown,
  Menu,
  Header,
  OverflowIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { PreviewIcon } from './PreviewIcon';

// Note: this button has a different size from normal AntD buttons.
export { Button as HeaderButton };

type OverflowMenuProps = {
  menu?: {
    id: string;
    itemName: React.ReactNode;
    onClick?: (...args: any[]) => any;
    href?: string;
  }[];
};

export function OverflowMenu({ menu }: OverflowMenuProps) {
  const overflowMenu = (
    <Menu>
      {/* @ts-expect-error TS(2532): Object is possibly 'undefined'. */}
      {menu.map(({ id, itemName, onClick, href, ...otherProps }) => (
        // @ts-expect-error TS(2769): No overload matches this call.
        <Menu.Item key={id} onClick={onClick} href={href} data-test-id={id} {...otherProps}>
          {itemName}
        </Menu.Item>
      ))}
    </Menu>
  );

  // @ts-expect-error TS(2532): Object is possibly 'undefined'.
  return menu.length > 0 ? (
    <Dropdown overlay={overflowMenu} trigger={['click']} placement='bottomLeft' arrow>
      <Button icon={<OverflowIcon />} data-test-id='overflow-menu-trigger' />
    </Dropdown>
  ) : null;
}

type PageHeaderProps = {
  title: React.ReactNode;
  breadcrumbs?: React.ReactNode[];
  preview?: boolean;
  feedbackForm?: string;
  infoPopover?: React.ReactNode;
  children?: React.ReactNode;
};

/**
 * A page header that includes:
 *   - title,
 *   - optional breadcrumb content,
 *   - optional preview mark,
 *   - optional feedback link, and
 *   - optional info popover, safe to have link inside.
 */
export function PageHeader(props: PageHeaderProps) {
  const {
    title, // required
    breadcrumbs = [],
    preview,
    children,
  } = props;
  const { theme } = useDesignSystemTheme();
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
            {preview && <PreviewIcon css={{ marginRight: theme.spacing.sm }}/>}
          </>
        }
      />
      <Spacer
        // @ts-expect-error TS(2322): Type '{ css: { flexShrink: number; }; }' is not as... Remove this comment to see the full error message
        css={{
          // Ensure spacer's fixed height
          flexShrink: 0,
        }}
      />
    </>
  );
}
