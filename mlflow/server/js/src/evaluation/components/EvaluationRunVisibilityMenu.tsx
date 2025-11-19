import React from 'react';
import { Button, Dropdown, Menu } from 'antd';

type Props = {
  onChangeVisibility: (mode: string) => void;
};

export const EvaluationRunVisibilityMenu: React.FC<Props> = ({ onChangeVisibility }) => {
  const menu = (
    <Menu
      onClick={(info) => {
        onChangeVisibility(info.key);
      }}
      items={[
        { label: 'Show all runs', key: 'show_all' },
        { label: 'Hide finished runs', key: 'hide_finished' },
        { label: 'Show first 10 runs', key: 'show_first_10' },
      ]}
    />
  );

  return (
    <Dropdown overlay={menu} trigger={['click']}>
      <Button>Visibility Options</Button>
    </Dropdown>
  );
};
