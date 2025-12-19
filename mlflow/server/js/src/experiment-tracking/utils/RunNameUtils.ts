/**
 * TS implementation of
 * https://github.com/mlflow/mlflow/blob/master/mlflow/utils/name_utils.py
 */

import { RUNS_COLOR_PALETTE } from '../../common/color-palette';
// prettier-ignore
const dictionaryAdjectives = ['abundant','able','abrasive','adorable','adaptable','adventurous','aged','agreeable','ambitious','amazing','amusing','angry','auspicious','awesome','bald','beautiful','bemused','bedecked','big','bittersweet','blushing','bold','bouncy','brawny','bright','burly','bustling','calm','capable','carefree','capricious','caring','casual','charming','chill','classy','clean','clumsy','colorful','crawling','dapper','debonair','dashing','defiant','delicate','delightful','dazzling','efficient','enchanting','entertaining','enthused','exultant','fearless','flawless','fortunate','fun','funny','gaudy','gentle','gifted','glamorous','grandiose','gregarious','handsome','hilarious','honorable','illustrious','incongruous','indecisive','industrious','intelligent','inquisitive','intrigued','invincible','judicious','kindly','languid','learned','legendary','likeable','loud','luminous','luxuriant','lyrical','magnificent','marvelous','masked','melodic','merciful','mercurial','monumental','mysterious','nebulous','nervous','nimble','nosy','omniscient','orderly','overjoyed','peaceful','painted','persistent','placid','polite','popular','powerful','puzzled','rambunctious','rare','rebellious','respected','resilient','righteous','receptive','redolent','resilient','rogue','rumbling','salty','sassy','secretive','selective','sedate','serious','shivering','skillful','sincere','skittish','silent','smiling','sneaky','sophisticated','spiffy','stately','suave','stylish','tasteful','thoughtful','thundering','traveling','treasured','trusting','unequaled','upset','unique','unleashed','useful','upbeat','unruly','valuable','vaunted','victorious','welcoming','whimsical','wistful','wise','worried','youthful','zealous'];
// prettier-ignore
const dictionaryNouns = ['ant','ape','asp','auk','bass','bat','bear','bee','bird','boar','bug','calf','carp','cat','chimp','cod','colt','conch','cow','crab','crane','croc','crow','cub','deer','doe','dog','dolphin','donkey','dove','duck','eel','elk','fawn','finch','fish','flea','fly','foal','fowl','fox','frog','gnat','gnu','goat','goose','grouse','grub','gull','hare','hawk','hen','hog','horse','hound','jay','kit','kite','koi','lamb','lark','loon','lynx','mare','midge','mink','mole','moose','moth','mouse','mule','newt','owl','ox','panda','penguin','perch','pig','pug','quail','ram','rat','ray','robin','roo','rook','seal','shad','shark','sheep','shoat','shrew','shrike','shrimp','skink','skunk','sloth','slug','smelt','snail','snake','snipe','sow','sponge','squid','squirrel','stag','steed','stoat','stork','swan','tern','toad','trout','turtle','vole','wasp','whale','wolf','worm','wren','yak','zebra'];

const generateString = (separator: string, integerScale: number) => {
  const randomAdjIndex = Math.floor(Math.random() * dictionaryAdjectives.length);
  const randomNounIndex = Math.floor(Math.random() * dictionaryNouns.length);
  const randomAdjective = dictionaryAdjectives[randomAdjIndex];
  const randomNoun = dictionaryNouns[randomNounIndex];
  const randomNumber = Math.floor(Math.random() * 10 ** integerScale);
  return [randomAdjective, randomNoun, randomNumber].join(separator);
};

/**
 * Generates a random name suitable for experiment run, e.g. invincible-mule-479
 */
export const generateRandomRunName = (separator = '-', integerScale = 3, maxLength = 20) => {
  let name = '';
  for (let i = 0; i < 10; i++) {
    name = generateString(separator, integerScale);
    if (name.length < maxLength) return name;
  }
  return name.slice(0, maxLength);
};

export const getDuplicatedRunName = (originalRunName = '', alreadyExistingRunNames: string[] = []) => {
  // Check if the the run name being copied is already suffixed with number
  const match = originalRunName.match(/\s\((\d+)\)$/);

  const nameSegmentWithoutIndex = match
    ? originalRunName.substring(0, originalRunName.length - match[0].length)
    : originalRunName;

  let newIndex = match ? parseInt(match[1], 10) + 1 : 1;
  // If there's already a run name with increased index, increase it again
  while (alreadyExistingRunNames.includes(nameSegmentWithoutIndex + ' (' + newIndex + ')')) {
    newIndex++;
  }
  return nameSegmentWithoutIndex + ' (' + newIndex + ')';
};

/**
 * Temporary function that assigns randomized, yet stable color
 * from the static palette basing on an input string. Used for coloring runs.
 *
 * TODO: make a decision on the final color hashing per run
 */
export const getStableColorForRun = (runUuid: string) => {
  let a = 0,
    b = 0;

  // Let's use super simple hashing method
  for (let i = 0; i < runUuid.length; i++) {
    a = (a + runUuid.charCodeAt(i)) % 255;
    b = (b + a) % 255;
  }

  // eslint-disable-next-line no-bitwise
  return RUNS_COLOR_PALETTE[(a | (b << 8)) % RUNS_COLOR_PALETTE.length];
};
