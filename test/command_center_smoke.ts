// Run against an active command center: bun test/command_center_smoke.ts [URL]
const base = process.argv[2] ?? "http://127.0.0.1:8000";
for (const [path, expected] of [["/", 200], ["/ui/study.css", 200], ["/ui/study.js", 200], ["/missing.html", 404], ["/%2e%2e/README.md", 404]] as const) {
  const response = await fetch(base + path);
  if (response.status !== expected) throw new Error(`${path}: ${response.status}, expected ${expected}`);
  if (path === "/") {
    const page = await response.text();
    if (!page.includes("Choose what to explore") || !page.includes("Compare solution representations") || page.includes("$date")) throw new Error("Report template was not applied");
  }
  console.log(`${path}: ${response.status}`);
}
export {};
