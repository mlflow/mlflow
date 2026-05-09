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
    `sudo tshark -r ${pcap} -Y 'tls.handshake.type==1' ` +
      `-T fields -e tls.handshake.extensions_server_name 2>/dev/null | sort -u`,
    { shell: "/bin/bash" }
  )
    .toString()
    .trim();
} catch {}

console.log("Outbound hosts contacted during this job:");
if (hosts) {
  for (const host of hosts.split("\n")) {
    console.log(`  ${host}`);
  }
} else {
  console.log("  (none captured)");
}
