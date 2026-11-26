import { describe, expect, it } from '@jest/globals';
import { hierarchyLineStyles, TimelineTreeZIndex } from './hierarchyLineStyles';

describe('hierarchy lines', () => {
  it('keeps enabled ancestor line positions and colors across gaps', () => {
    expect(
      hierarchyLineStyles(
        [
          { shouldRender: true, isActive: false },
          { shouldRender: false, isActive: true },
          { shouldRender: true, isActive: true },
        ],
        'blue',
        'grey',
      ),
    ).toEqual({
      position: 'absolute',
      inset: 0,
      backgroundImage: 'linear-gradient(grey, grey), linear-gradient(blue, blue)',
      backgroundPosition: '8px 0, 40px 0',
      backgroundSize: '1px 100%',
      backgroundRepeat: 'no-repeat',
      zIndex: TimelineTreeZIndex.NORMAL,
    });
  });
  it.each([
    { shouldRender: false, isActive: false },
    { shouldRender: true, isActive: false },
    { shouldRender: false, isActive: true },
  ])('keeps an inactive or absent last line beneath the connector: %p', (last) => {
    expect(hierarchyLineStyles([{ shouldRender: true, isActive: true }, last], 'blue', 'grey')?.zIndex).toBe(
      TimelineTreeZIndex.LOW,
    );
  });
  it('omits the line layer when no straight lines are enabled', () => {
    expect(hierarchyLineStyles([], 'blue', 'grey')).toBeUndefined();
    expect(hierarchyLineStyles([{ shouldRender: false, isActive: false }], 'blue', 'grey')).toBeUndefined();
  });
});
