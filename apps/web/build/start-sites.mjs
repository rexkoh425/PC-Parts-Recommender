import path from "node:path";
import { fileURLToPath } from "node:url";

const arguments_ = process.argv.slice(2);

function optionValue(...names) {
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    for (const name of names) {
      if (argument === name) {
        return arguments_[index + 1];
      }
      if (argument.startsWith(`${name}=`)) {
        return argument.slice(name.length + 1);
      }
    }
  }
  return undefined;
}

const portValue = optionValue("--port", "-p") ?? process.env.PORT ?? "3000";
const port = Number.parseInt(portValue, 10);
if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new Error(`Invalid Sites server port: ${portValue}`);
}

const host = optionValue("--hostname", "--host", "-H") ?? "0.0.0.0";
const outDir = fileURLToPath(new URL("../dist", import.meta.url));
const vinextServerUrl = new URL("../node_modules/vinext/dist/server/prod-server.js", import.meta.url);

// vinext 0.0.50 builds its static-file cache from path.relative(). On Windows,
// that produces backslash-delimited cache keys which cannot match URL paths.
// Normalize only while the startup cache is populated, then restore Node's
// implementation. The deployed Cloudflare worker does not use this shim.
const originalRelative = path.relative;
if (process.platform === "win32") {
  path.relative = (...values) => originalRelative(...values).replaceAll(path.sep, "/");
}

try {
  const { startProdServer } = await import(vinextServerUrl.href);
  await startProdServer({ host, port, outDir });
} finally {
  path.relative = originalRelative;
}
