import { defineConfig, devices } from "@playwright/test";

const port = 3200;
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e-sites",
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  reporter: "list",
  use: {
    baseURL,
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: {
    command: `npm run start:sites -- --hostname 127.0.0.1 --port ${port}`,
    env: {
      NEXT_PUBLIC_DATA_MODE: "demo",
      NEXT_PUBLIC_SITE_URL: baseURL,
      NODE_ENV: "production",
    },
    url: baseURL,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
