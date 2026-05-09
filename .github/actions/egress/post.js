const { execSync } = require("node:child_process");

const pid = process.env.STATE_pid;
const pcap = process.env.STATE_pcap;

try {
  execSync(`sudo kill -INT ${pid}`, { stdio: "ignore" });
} catch {}
execSync("sleep 1");

let hosts = "";
try {
  hosts = execSync(
    `sudo tcpdump -r ${pcap} -nn -A 'tcp dst port 443' ` +
      `| grep -oE '[a-z0-9][a-z0-9.-]*\\.[a-z]{2,63}' | sort -u`,
    { shell: "/bin/bash" }
  )
    .toString()
    .trim();
} catch (err) {
  console.log(`extraction failed: ${err.message}`);
}

console.log("Outbound hosts contacted during this job:");
if (hosts) {
  for (const host of hosts.split("\n")) {
    console.log(`  ${host}`);
  }
} else {
  console.log("  (none captured)");
}
