// Script to generate aliases for icons using ChatGPT
// Script will automatically generate aliases for new icons.

// Pre-requisites:
// 1. Install inkscape cli tool
// 2. Get API key from OpenAI
// 3. Set API key in environment variable API_KEY
// 4. Run the script

const exec = require('child_process').exec;
const fs = require('fs');
const path = require('path');

// eslint-disable-next-line import/no-extraneous-dependencies
const axios = require('axios');

let existingAliases = require('./../../../assets/icons/aliases.json');

const iconFolder = path.resolve(__dirname, '../../../assets/icons');

const files = fs.readdirSync(iconFolder);
const icons = files.map((file) => path.parse(file)).filter((file) => file.ext === '.svg');

const newIcons = icons.filter((icon) => !existingAliases[icon.name]);

// eslint-disable-next-line no-console -- TODO(FEINF-3587)
console.log('Handling new icons:', newIcons.length);

(async () => {
  for (const icon of newIcons) {
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    console.log('Processing', icon.name);
    await new Promise((resolve, reject) => {
      exec(`inkscape -w 128 -h 128 ../../../assets/icons/${icon.name}.svg -o ${icon.name}.png`, (err, stdout) => {
        if (err) {
          // eslint-disable-next-line no-console -- TODO(FEINF-3587)
          console.error('Error:', err);
          reject(err);
        }
        resolve(stdout);
      });
    });

    const generatedAliases = await askChatGPT({ name: icon.name, imagePath: `${icon.name}.png` });

    existingAliases = {
      ...existingAliases,
      [icon.name]: generatedAliases,
    };
    fs.writeFileSync('./../../../assets/icons/aliases.json', JSON.stringify(existingAliases, null, 4));
    // Uncomment the below line to avoid rate
    // await new Promise((resolve) => setTimeout(resolve, 1000));
  }
})();

async function askChatGPT({ name, imagePath }) {
  // OpenAI API Key
  const apiKey = process.env.API_KEY;
  // Function to encode the image to base64
  const encodeImage = (imagePath) => {
    const image = fs.readFileSync(imagePath);
    return image.toString('base64');
  };

  // Getting the base64 string
  const base64Image = encodeImage(imagePath);

  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
  };

  const payload = {
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Icon has a name: ${name}. Get possible aliases for the image in a JSON array format, do not include Icon name in the end of a string.`,
          },
          {
            type: 'image_url',
            image_url: {
              url: `data:image/jpeg;base64,${base64Image}`,
            },
          },
        ],
      },
    ],
  };

  // Make the POST request using axios
  const response = await axios.post('https://api.openai.com/v1/chat/completions', payload, { headers });

  const aliases = JSON.parse(response.data.choices[0].message.content.replace('```json', '').replace('```', ''));
  return aliases;
}
