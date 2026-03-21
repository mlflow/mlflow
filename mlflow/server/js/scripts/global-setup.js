const { execSync } = require('child_process');
const os = require('os');

module.exports = () => {
  // On windows, the timezone is not set with `TZ=GMT`. As a workaround, use `tzutil`.
  // This approach is taken from https://www.npmjs.com/package/set-tz.
  if (os.platform() === 'win32') {
    const TZ = 'UTC';
    const previousTZ = execSync('tzutil /g').toString();
    const cleanup = () => {
      execSync(`tzutil /s "${previousTZ}"`);
      console.log(`Restored timezone to ${previousTZ}`);
    };
    execSync(`tzutil /s "${TZ}"`);
    console.warn(
      `Changed timezone to ${TZ}. If process is killed, manually run: tzutil /s "${previousTZ}"`,
    );
    process.on('exit', cleanup);
    process.on('SIGINT', () => {
      process.exit(2);
    });
  } else {
    process.env.TZ = 'GMT';
  }
};
