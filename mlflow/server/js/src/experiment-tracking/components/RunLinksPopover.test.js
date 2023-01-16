import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { shallow, mount } from 'enzyme';

import { RunLinksPopover } from './RunLinksPopover';
import Routes from '../routes';

describe('unit tests', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      experimentId: '0',
      visible: false,
      x: 0,
      y: 0,
      runItems: [],
      handleClose: jest.fn(),
      handleKeyDown: jest.fn(),
      handleVisibleChange: jest.fn(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<RunLinksPopover {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render two links when two run items are given', () => {
    const props = {
      ...minimalProps,
      visible: true,
      runItems: [
        {
          runId: 'runUuid1',
          name: 'run1',
          color: 'rgb(1, 1, 1)',
          y: 0.1,
        },
        {
          runId: 'runUuid2',
          name: 'run2',
          color: 'rgb(2, 2, 2)',
          y: 0.2,
        },
      ],
    };

    wrapper = mount(
      <MemoryRouter>
        <RunLinksPopover {...props} />
      </MemoryRouter>,
    ).find(RunLinksPopover);

    // The popover is attached to the document root and can't be found with wrapper.find.
    const popover = document.getElementsByClassName('ant-popover')[0];
    const links = popover.querySelectorAll('a[href]');
    expect(links.length).toBe(2);

    props.runItems.forEach(({ runId, name, color, y }, index) => {
      const link = links[index];
      const hrefExpected = Routes.getRunPageRoute(props.experimentId, runId);
      expect(link.getAttribute('href')).toBe(hrefExpected);

      const p = link.querySelector('p');
      expect(p.textContent).toBe(`${name}, ${y}`);
      expect(p.style.color).toBe(color);
    });
  });
});
