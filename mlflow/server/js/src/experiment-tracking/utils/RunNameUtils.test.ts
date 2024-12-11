import { createPrng } from '../../common/utils/TestUtils';
import { generateRandomRunName, getDuplicatedRunName } from './RunNameUtils';

describe('RunNameUtils', () => {
  beforeEach(() => {
    jest.spyOn(global.Math, 'random').mockImplementation(createPrng());
  });
  afterEach(() => {
    jest.spyOn(global.Math, 'random').mockRestore();
  });

  it.each([5, 10, 30])(
    'correctly generate random run names with respect to the maximum length of %s characters',
    (maxLength) => {
      const generatedNames = new Array(10).fill(0).map(() => generateRandomRunName('-', 3, maxLength));

      for (const name of generatedNames) {
        expect(name.length).toBeLessThanOrEqual(maxLength);
      }
    },
  );

  describe('getCopiedRunName', () => {
    it('creates simple copied name', () => {
      expect(getDuplicatedRunName('run-name')).toEqual('run-name (1)');
    });

    it('creates duplicated name out of already duplicated name', () => {
      expect(getDuplicatedRunName('run-name (1)')).toEqual('run-name (2)');
      expect(getDuplicatedRunName('run-name (2)')).toEqual('run-name (3)');
      expect(getDuplicatedRunName('run-name (9)')).toEqual('run-name (10)');
      expect(getDuplicatedRunName('run-name (99)')).toEqual('run-name (100)');
    });

    it('creates duplicated name considering already existing names', () => {
      expect(getDuplicatedRunName('run-name', ['run-name (1)'])).toEqual('run-name (2)');
      expect(getDuplicatedRunName('run-name (1)', ['run-name (2)'])).toEqual('run-name (3)');
      expect(getDuplicatedRunName('run-name (17)', ['run-name (18)', 'run-name (19)'])).toEqual('run-name (20)');
      expect(getDuplicatedRunName('run-name (15)', ['run-name (10)'])).toEqual('run-name (16)');
    });
  });
});
