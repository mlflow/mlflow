// eslint-disable-next-line @databricks/no-restricted-imports-regexp
import type { ReactWrapper } from 'enzyme';

export * as selectEvent from './selectEvent';

export * from './table';

/**
 * Open a dropdown menu by simulating a pointerDown event on the dropdown button.
 *
 * @param dropdownButton - The Dropdown Trigger button that opens the menu when clicked.
 */
export const openDropdownMenu = <P, S, C>(dropdownButton: ReactWrapper<P, S, C>) => {
  dropdownButton.hostNodes().simulate('pointerDown', { button: 0, ctrlKey: false });
};
