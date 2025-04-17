import { Dash } from 'plotly.js';
import { useMemo, useState } from 'react';
import { IntlProvider } from 'react-intl';

/**
 * Creates a stable (seeded) function that returns
 * gaussian-distributed randomized values
 */
export const stableNormalRandom = (seed = 0, g = 10) => {
  const random = () => {
    // eslint-disable-next-line no-param-reassign
    seed += 0x6d2b79f5;
    let t = seed;
    // eslint-disable-next-line no-bitwise
    t = Math.imul(t ^ (t >>> 15), t | 1);
    // eslint-disable-next-line no-bitwise
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    // eslint-disable-next-line no-bitwise
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => {
    let sum = 0;

    for (let i = 0; i < g; i += 1) {
      sum += random();
    }

    return sum / g;
  };
};

// An array of sample colors
// prettier-ignore
export const chartColors = ["#CE4661", "#3183AF", "#B65F95", "#FACE76", "#A96F2F", "#458C32", "#697A88", "#76AFCD", "#8F6DBF", "#C84856"];

// Arrays of sample adjectives and animal names
// prettier-ignore
const adjectives = ["ablaze","abrupt","accomplished","active","adored","adulated","adventurous","affectionate","amused","amusing","animal-like","antique","appreciated","archaic","ardent","arrogant","astonished","audacious","authoritative","awestruck","beaming","bewildered","bewitching","blissful","boisterous","booming","bouncy","breathtaking","bright","brilliant","bubbling","calm","calming","capricious","celestial","charming","cheerful","cherished","chiaroscuro","chilled","comical","commanding","companionable","confident","contentment","courage","crazy","creepy","dancing","dazzling","delicate","delightful","demented","desirable","determined","devoted","dominant","dramatic","drawn out","dripping","dumbstruck","ebullient","elated","expectant","expressive","exuberant","faint","fantastical","favorable","febrile","feral","feverish","fiery","floating","flying","folksy","fond","forgiven","forgiving","freakin' awesome","frenetic","frenzied","friendly. amorous","from a distance","frosted","funny","furry","galloping","gaping","gentle","giddy","glacial","gladness","gleaming","gleeful","gorgeous","imperious","impudent","in charge","inflated","innocent","inspired","intimate","intrepid","jagged","joking","joyful","jubilant","kindly","languid","larger than life","laughable","lickety-split","lighthearted","limping","linear","lively","lofty","love of","lovely","lulling","luminescent","lush","luxurious","magical","maniacal","manliness","march-like","masterful","merciful","merry","mischievous","misty","modest","moonlit","mysterious","mystical","mythological","nebulous","nostalgic","on fire","overstated","paganish","partying","perfunctory","perky","perplexed","persevering","pious","playful","pleasurable","poignant","portentous","posh","powerful","pretty","prickly","prideful","princesslike","proud","puzzled","queenly","questing","quiet","racy","ragged","regal","rejoicing","relaxed","reminiscent","repentant","reserved","resolute","ridiculous","ritualistic","robust","running","sarcastic","scampering","scoffing","scurrying","sensitive","serene","shaking","shining","silky","silly","simple","successful","summery","surprised","sympathetic","tapping","virile","walking","wild","witty","wondering","zealous","zestful"];
// prettier-ignore
const animalNames = ["aardvark","albatross","alligator","alpaca","ant","anteater","antelope","ape","armadillo","donkey","baboon","badger","barracuda","bat","bear","beaver","bee","bison","boar","buffalo","butterfly","camel","capybara","caribou","cassowary","cat","caterpillar","cattle","chamois","cheetah","chicken","chimpanzee","chinchilla","chough","clam","cobra","cockroach","cod","cormorant","coyote","crab","crane","crocodile","crow","curlew","deer","dinosaur","dog","dogfish","dolphin","dotterel","dove","dragonfly","duck","dugong","dunlin","eagle","echidna","eel","eland","elephant","elk","emu","falcon","ferret","finch","fish","flamingo","fly","fox","frog","gaur","gazelle","gerbil","giraffe","gnat","gnu","goat","goldfinch","goldfish","goose","gorilla","goshawk","grasshopper","grouse","guanaco","gull","hamster","hare","hawk","hedgehog","heron","herring","hippopotamus","hornet","horse","human","hummingbird","hyena","ibex","ibis","jackal","jaguar","jay","jellyfish","kangaroo","kingfisher","koala","kookabura","kouprey","kudu","lapwing","lark","lemur","leopard","lion","llama","lobster","locust","loris","louse","lyrebird","magpie","mallard","manatee","mandrill","mantis","marten","meerkat","mink","mole","mongoose","monkey","moose","mosquito","mouse","mule","narwhal","newt","nightingale","octopus","okapi","opossum","oryx","ostrich","otter","owl","oyster","panther","parrot","partridge","peafowl","pelican","penguin","pheasant","pig","pigeon","pony","porcupine","porpoise","quail","quelea","quetzal","rabbit","raccoon","rail","ram","rat","raven","red deer","red panda","reindeer","rhinoceros","rook","salamander","salmon","sand dollar","sandpiper","sardine","scorpion","seahorse","seal","shark","sheep","shrew","skunk","snail","snake","sparrow","spider","spoonbill","squid","squirrel","starling","stingray","stinkbug","stork","swallow","swan","tapir","tarsier","termite","tiger","toad","trout","turkey","turtle","viper","vulture","wallaby","walrus","wasp","weasel","whale","wildcat","wolf","wolverine","wombat","woodcock","woodpecker","worm","wren","yak","zebra"];

export const getRandomRunName = (randomFn = Math.random) =>
  `${adjectives[Math.floor(randomFn() * adjectives.length)]}-${
    animalNames[Math.floor(randomFn() * animalNames.length)]
  }-${Math.floor(randomFn() * 1000)}`;

export const ChartStoryWrapper = ({ children, controls }: React.PropsWithChildren<any>) => (
  <IntlProvider locale="en">
    <div
      css={{
        width: '100vw',
        height: '100vh',
        padding: 20,
        display: 'grid',
        gridTemplateRows: 'auto 1fr',
      }}
    >
      <div
        css={{
          display: 'flex',
          gap: 16,
          alignItems: 'center',
          marginBottom: 16,
          backgroundColor: '#eee',
          padding: 8,
        }}
      >
        {controls || null}
      </div>
      {children}
    </div>
  </IntlProvider>
);

export const useControls = (zAxisVisible = false) => {
  const [xKey, setXKey] = useState('metric1');
  const [yKey, setYKey] = useState('param2');
  const [zKey, setZKey] = useState('param3');

  const axisProps = useMemo(
    () => ({
      xAxis: {
        key: xKey,
        type: xKey.includes('metric') ? ('METRIC' as const) : ('PARAM' as const),
      },
      yAxis: {
        key: yKey,
        type: yKey.includes('metric') ? ('METRIC' as const) : ('PARAM' as const),
      },
      zAxis: {
        key: zKey,
        type: zKey.includes('metric') ? ('METRIC' as const) : ('PARAM' as const),
      },
    }),
    [xKey, yKey, zKey],
  );

  const getOptions = () => (
    <>
      <option>metric1</option>
      <option>metric2</option>
      <option>metric3</option>
      <option>param1</option>
      <option>param2</option>
      <option>param3</option>
    </>
  );

  const controls = (
    <>
      X axis:{' '}
      <select value={xKey} onChange={({ target }) => setXKey(target.value)}>
        {getOptions()}
      </select>{' '}
      Y axis:{' '}
      <select value={yKey} onChange={({ target }) => setYKey(target.value)}>
        {getOptions()}
      </select>{' '}
      {zAxisVisible ? (
        <>
          Z axis:{' '}
          <select value={zKey} onChange={({ target }) => setZKey(target.value)}>
            {getOptions()}
          </select>
        </>
      ) : null}
    </>
  );

  return {
    axisProps,
    controls,
  };
};
