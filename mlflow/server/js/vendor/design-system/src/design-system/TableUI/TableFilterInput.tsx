import { css } from '@emotion/react';
import type { Input as AntDInput } from 'antd';
import { forwardRef } from 'react';

import type { Theme } from '../../theme';
import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks';
import { SearchIcon } from '../Icon';
import type { InputProps } from '../Input';
import { Input } from '../Input';
import type { HTMLDataAttributes } from '../types';

interface TableFilterInputProps extends InputProps, HTMLDataAttributes {
  // Called when the form is submitted, either by pressing enter or clicking the search button.
  // If provided, this will cause the input to be rendered within a form.
  onSubmit?: () => void;
  // Whether to show the search button.
  showSearchButton?: boolean;
  searchButtonProps?: Omit<
    React.ComponentProps<typeof Button>,
    'children' | 'type' | 'size' | 'componentId' | 'analyticsEvents'
  >;
  containerProps?: React.ComponentProps<'div'>;
}

const getTableFilterInputStyles = (theme: Theme, defaultWidth: number) => {
  return css({
    [theme.responsive.mediaQueries.sm]: {
      width: 'auto',
    },

    [theme.responsive.mediaQueries.lg]: {
      width: '30%',
    },

    [theme.responsive.mediaQueries.xxl]: {
      width: defaultWidth,
    },
  });
};

export const TableFilterInput = forwardRef<AntDInput, TableFilterInputProps>(function SearchInput(
  { onSubmit, showSearchButton, className, containerProps, searchButtonProps, ...inputProps },
  ref,
) {
  const { theme } = useDesignSystemTheme();
  const DEFAULT_WIDTH = 400;

  let component = <Input prefix={<SearchIcon />} allowClear {...inputProps} className={className} ref={ref} />;

  if (showSearchButton) {
    component = (
      <Input.Group
        css={{
          display: 'flex',
          width: '100%',
        }}
        className={className}
      >
        <Input
          allowClear
          {...inputProps}
          ref={ref}
          css={{
            flex: 1,
          }}
        />
        <Button
          componentId={
            inputProps.componentId
              ? `${inputProps.componentId}.search_submit`
              : 'codegen_design-system_src_design-system_tableui_tablefilterinput.tsx_65'
          }
          htmlType="submit"
          aria-label="Search"
          {...searchButtonProps}
        >
          <SearchIcon />
        </Button>
      </Input.Group>
    );
  }

  return (
    <div
      style={{
        height: theme.general.heightSm,
      }}
      css={getTableFilterInputStyles(theme, DEFAULT_WIDTH)}
      {...containerProps}
    >
      {onSubmit ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit();
          }}
        >
          {component}
        </form>
      ) : (
        component
      )}
    </div>
  );
});
