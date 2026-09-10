import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 60_000,
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure'
  },
  webServer: {
    command: 'npm run dev -- --host 0.0.0.0',
    url: 'http://localhost:3000',
    reuseExistingServer: true,
    timeout: 30_000
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }]
});