import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { shallow, mount } from 'enzyme';

import { RunLinksPopover } from './RunLinksPopover';
import Routes from '../Routes';

describe('unit tests', () => {
  let wrapper;
  const minimumProps = { experimentId: 0 };

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<RunLinksPopover {...minimumProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render two run links when two run items are given', () => {
    wrapper = mount(
      <MemoryRouter>
        <RunLinksPopover {...minimumProps} />
      </MemoryRouter>
    ).find(RunLinksPopover);

    const state = {
      visible: true,
      runItems: [
        {
          name: 'run1',
          color: 'rgb(1, 1, 1)',
          runUuid: 'runUuid1',
        },
        {
          name: 'run2',
          color: 'rgb(2, 2, 2)',
          runUuid: 'runUuid2',
        },
      ],
    };

    wrapper.instance().setState(state);
    // The popover is attached to the document root and can't be found with wrapper.find.
    const popover = document.getElementsByClassName('ant-popover')[0];
    const links = popover.querySelectorAll('a[href]');
    expect(links.length).toBe(2);

    state.runItems.forEach(({ name, runUuid, color }, index) => {
      const link = links[index];
      const hrefExpected = Routes.getRunPageRoute(minimumProps.experimentId, runUuid);
      expect(link.getAttribute('href')).toBe(hrefExpected);

      const p = link.querySelector('p');
      expect(p.textContent).toBe(name);
      expect(p.style.color).toBe(color);
    });
  });
});
