const { exec } = require("child_process");

exec("git ls-files --others --exclude-standard", (error, stdout, stderr) => {
  if (error) {
    console.error(`An error occurred: ${error}`);
    process.exit(error.code || 1);
  }

  const untrackedFiles = stdout.trim();

  if (untrackedFiles === "") {
    console.log("No untracked files found.");
    process.exit(0);
  } else {
    console.log("Untracked files found:");
    console.log(untrackedFiles);
    console.log("Consider adding them to .gitignore.");
    process.exit(1);
  }
});
