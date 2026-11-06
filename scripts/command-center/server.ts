import { readdir, realpath } from "node:fs/promises";
import { resolve, sep } from "node:path";

const root = resolve(import.meta.dir, "../../studies/webui");
const ui = resolve(import.meta.dir, "../charts/ui");
const args = process.argv.slice(2);
if (args.includes("--help")) {
  console.log("bun run command-center [--port 8000]\nServes saved studies locally. Export new experiment results separately.");
  process.exit(0);
}
if (args.some((arg, i) => arg !== "--port" && args[i - 1] !== "--port")) {
  throw new Error("Unknown argument. Use --help for usage.");
}
const port = Number(args.includes("--port") ? args[args.indexOf("--port") + 1] : process.env.PORT ?? 8000);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Port must be between 1 and 65535.");
const build = await Bun.build({entrypoints: [resolve(ui, "bootstrap.ts")], target: "browser", format: "iife"});
if (!build.success) throw new AggregateError(build.logs, "Frontend build failed");
const script = await build.outputs[0].text();
const template = await Bun.file(resolve(ui, "study.html")).text();
const controls = template.match(/<div id="study-controls"[\s\S]*?(?=<div id="comparison-key")/)?.[0];
const header = template.match(/<header>[\s\S]*?<\/header>/)?.[0];
const server = Bun.serve({
  hostname: "127.0.0.1", port,
  async fetch(request) {
    const url = new URL(request.url);
    if (request.method !== "GET" && request.method !== "HEAD") return new Response("Method not allowed", {status: 405});
    if (url.pathname === "/") {
      const reports = (await readdir(root)).filter(name => /^dual-representation-[\d-]+\.html$/.test(name)).sort();
      if (!reports.length) return new Response("No saved studies found. Export experiment results to studies/webui, then refresh.", {status: 404});
      return Response.redirect(new URL(reports.at(-1)!, url), 302);
    }
    if (url.pathname === "/ui/study.js") return new Response(script, {headers: {"Content-Type": "text/javascript", "Cache-Control": "no-store"}});
    if (url.pathname === "/ui/study.css") return new Response(Bun.file(resolve(ui, "study.css")), {headers: {"Cache-Control": "no-store"}});
    let path: string;
    try {
      path = await realpath(resolve(root, "." + decodeURIComponent(url.pathname)));
    } catch { return new Response("Not found", {status: 404}); }
    if (!path.startsWith(root + sep)) return new Response("Not found", {status: 404});
    if (!/\.(html|json|png|pdf|csv)$/.test(path)) return new Response("Not found", {status: 404});
    const file = Bun.file(path);
    if (path.endsWith(".html")) {
      let page = await file.text();
      const date = page.match(/\d{4}-\d{2}-\d{2}/)?.[0] ?? "Saved study";
      const count = page.match(/(\d+) (?:paired|comparison) figures/)?.[1] ?? "—";
      if (header) page = page.replace(/<header>[\s\S]*?<\/header>/, header.replaceAll("$date", date).replaceAll("$figure_count", count));
      if (controls) page = page.replace(/<div id="study-controls"[\s\S]*?(?=<div id="comparison-key")/, controls);
      return new Response(page, {headers: {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"}});
    }
    return new Response(file);
  },
});
console.log(`PINNs command center: ${server.url}\nViewing saved results. Press Ctrl+C to stop.`);
