import React from 'react';
import { shallow } from 'enzyme';
import { ErrorView } from './ErrorView';

describe('ErrorView', () => {
  test('should render 400', () => {
    const wrapper = shallow(<ErrorView statusCode={400} fallbackHomePageReactRoute={'/path/to'} />);
    const errorImage = wrapper.childAt(0);
    const title = wrapper.childAt(1);
    const subtitle = wrapper.childAt(2);

    expect(errorImage.name()).toBe('ErrorImage');
    expect(errorImage.prop('statusCode')).toBe(400);
    expect(title.name()).toBe('h1');
    expect(title.text()).toBe('Bad Request');
    expect(subtitle.name()).toBe('h2');
    expect(subtitle.text()).toMatch(/^Go back to/);
    expect(
      subtitle
        .find('ForwardRef')
        .first()
        .prop('to'),
    ).toBe('/path/to');
  });

  it('should render 404', () => {
    const wrapper = shallow(<ErrorView statusCode={404} fallbackHomePageReactRoute={'/path/to'} />);

    const errorImage = wrapper.childAt(0);
    const title = wrapper.childAt(1);
    const subtitle = wrapper.childAt(2);

    expect(errorImage.name()).toBe('ErrorImage');
    expect(errorImage.prop('statusCode')).toBe(404);
    expect(title.name()).toBe('h1');
    expect(title.text()).toBe('Page Not Found');
    expect(subtitle.name()).toBe('h2');
    expect(subtitle.text()).toMatch(/^Go back to/);
    expect(
      subtitle
        .find('ForwardRef')
        .first()
        .prop('to'),
    ).toBe('/path/to');
  });

  test('should render 404 with sub message', () => {
    const wrapper = shallow(
      <ErrorView
        statusCode={404}
        fallbackHomePageReactRoute={'/path/to'}
        subMessage={'sub message'}
      />,
    );
    const errorImage = wrapper.childAt(0);
    const title = wrapper.childAt(1);
    const subtitle = wrapper.childAt(2);

    expect(errorImage.name()).toBe('ErrorImage');
    expect(errorImage.prop('statusCode')).toBe(404);
    expect(title.name()).toBe('h1');
    expect(title.text()).toBe('Page Not Found');
    expect(subtitle.name()).toBe('h2');
    expect(
      subtitle
        .find('ForwardRef')
        .first()
        .prop('to'),
    ).toBe('/path/to');
    expect(subtitle.text().split(', ')[0]).toBe('sub message');
    expect(subtitle.text().split(', ')[1]).toMatch(/^go back to/);
  });
});
