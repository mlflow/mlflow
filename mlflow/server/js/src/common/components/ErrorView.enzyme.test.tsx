import React from 'react';
import { ErrorView } from './ErrorView';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { MemoryRouter } from '../../common/utils/RoutingUtils';

describe('ErrorView', () => {
  test('should render 400', () => {
    const wrapper = mountWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={400} fallbackHomePageReactRoute="/path/to" />
      </MemoryRouter>,
    );
    const div = wrapper.childAt(0).childAt(0).childAt(0).childAt(0);

    const errorImage = div.childAt(0);
    const title = div.childAt(1);
    const subtitle = div.childAt(2);

    expect(errorImage.name()).toBe('ErrorImage');
    expect(errorImage.prop('statusCode')).toBe(400);
    expect(title.name()).toBe('h1');
    expect(title.text()).toBe('Bad Request');
    expect(subtitle.name()).toBe('h2');
    expect(subtitle.childAt(0).prop('defaultMessage')).toMatch(/^Go back to/);
    expect(subtitle.childAt(0).render().toString()).toMatch('/path/to');
  });

  it('should render 404', () => {
    const wrapper = mountWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={404} fallbackHomePageReactRoute="/path/to" />
      </MemoryRouter>,
    );
    const div = wrapper.childAt(0).childAt(0).childAt(0).childAt(0);

    const errorImage = div.childAt(0);
    const title = div.childAt(1);
    const subtitle = div.childAt(2);

    expect(errorImage.name()).toBe('ErrorImage');
    expect(errorImage.prop('statusCode')).toBe(404);
    expect(title.name()).toBe('h1');
    expect(title.text()).toBe('Page Not Found');
    expect(subtitle.name()).toBe('h2');
    expect(subtitle.childAt(0).prop('defaultMessage')).toMatch(/^Go back to/);
    expect(subtitle.childAt(0).render().toString()).toMatch('/path/to');
  });

  test('should render 404 with sub message', () => {
    const wrapper = mountWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={404} fallbackHomePageReactRoute="/path/to" subMessage="sub message" />
      </MemoryRouter>,
    );
    const div = wrapper.childAt(0).childAt(0).childAt(0).childAt(0);

    const errorImage = div.childAt(0);
    const title = div.childAt(1);
    const subtitle = div.childAt(2);

    expect(errorImage.name()).toBe('ErrorImage');
    expect(errorImage.prop('statusCode')).toBe(404);
    expect(title.name()).toBe('h1');
    expect(title.text()).toBe('Page Not Found');
    expect(subtitle.name()).toBe('h2');
    expect(subtitle.childAt(0).render().toString()).toMatch('sub message, go back to ');
    expect(subtitle.childAt(0).render().toString()).toMatch('/path/to');
  });
});
