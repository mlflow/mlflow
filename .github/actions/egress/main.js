const { spawn } = require("node:child_process");
const fs = require("node:fs");

const PCAP = "/tmp/egress.pcap";
const FILTER = "tcp and (port 80 or port 443) and not host 168.63.129.16";

const child = spawn("sudo", ["tcpdump", "-i", "any", "-U", "-w", PCAP, FILTER], {
  detached: true,
  stdio: "ignore",
});
child.unref();

fs.appendFileSync(process.env.GITHUB_STATE, `pid=${child.pid}\npcap=${PCAP}\n`);
console.log(`tcpdump started (sudo pid ${child.pid}), writing to ${PCAP}`);
