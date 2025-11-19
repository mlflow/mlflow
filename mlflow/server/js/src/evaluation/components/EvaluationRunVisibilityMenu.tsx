import React from 'react';
import { Dropdown, MenuProps, Button } from 'antd';

export type EvaluationVisibilityMode =
  | 'show_all'
  | 'hide_finished'
  | 'show_first_10';

type Props = {
  onChangeVisibility: (mode: EvaluationVisibilityMode) => void;
};

export const EvaluationRunVisibilityMenu: React.FC<Props> = ({ onChangeVisibility }) => {
  const items: MenuProps['items'] = [
    { key: 'show_all', label: 'Show all runs' },
    { key: 'hide_finished', label: 'Hide finished runs' },
    { key: 'show_first_10', label: 'Show first 10 runs' },
  ];

  const handleClick: MenuProps['onClick'] = (info) => {
    onChangeVisibility(info.key as EvaluationVisibilityMode);
  };

  return (
    <Dropdown
      menu={{ items, onClick: handleClick }}
      trigger={['click']}
    >
      <Button>Visibility Options</Button>
    </Dropdown>
  );
};

