/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { shallow, mount } from 'enzyme';

import { RunLinksPopover } from './RunLinksPopover';
import Routes from '../routes';

describe('unit tests', () => {
  let wrapper;
  let minimalProps: any;

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

    props.runItems.forEach(({ runId, name, color, y }: any, index: any) => {
      const link = links[index];
      const hrefExpected = Routes.getRunPageRoute(props.experimentId, runId);
      expect(link.getAttribute('href')).toBe(hrefExpected);

      const p = link.querySelector('p');
      // @ts-expect-error TS(2531): Object is possibly 'null'.
      expect(p.textContent).toBe(`${name}, ${y}`);
      // @ts-expect-error TS(2531): Object is possibly 'null'.
      expect(p.style.color).toBe(color);
    });
  });
});
