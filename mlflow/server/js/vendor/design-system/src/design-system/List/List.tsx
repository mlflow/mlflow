import type { ListProps as AntDListProps } from 'antd';
import { List as AntDList } from 'antd';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface ListProps<T> extends AntDListProps<T> {}

export const List = /* #__PURE__ */ (() => {
  function List<T>({ ...props }: ListProps<T>) {
    return (
      <DesignSystemAntDConfigProvider>
        <AntDList {...addDebugOutlineIfEnabled()} {...props} />
      </DesignSystemAntDConfigProvider>
    );
  }

  List.Item = AntDList.Item;

  return List;
})();
