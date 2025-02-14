import { render, act } from '@testing-library/react';

import { LegacyTabs } from './index';

describe('Tabs component', () => {
  let consoleErrorSpy: jest.SpyInstance;

  beforeEach(() => {
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  it('does not throw console warnings about unrecognized getPopupContainer prop', () => {
    act(() => {
      render(<LegacyTabs />);
    });

    expect(consoleErrorSpy).not.toHaveBeenCalledWith(
      expect.stringMatching(/Warning: React does not recognize the (.*) prop on a DOM element./),
      'getPopupContainer',
      'getpopupcontainer',
      expect.stringContaining('at Tabs'),
    );
  });
});
