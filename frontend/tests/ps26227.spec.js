import { test, expect } from '@playwright/test';

test('analyst search, map confidence markers, viewer slider, and review confirmation', async ({ page }) => {
  const searchResponse = page.waitForResponse(response =>
    response.url().includes('/search/text') && response.request().method() === 'POST'
  );

  await page.goto('/');
  const searchInput = page.getByPlaceholder(/Enter natural-language query/);
  await searchInput.fill('newly built structures near river');
  await page.getByRole('button', { name: /^Search$/ }).click();
  const response = await searchResponse;
  expect(response.status()).toBe(200);

  await expect(page.getByText(/Ranked Candidates/)).toBeVisible();
  await expect.poll(async () => page.locator('.maplibregl-marker').count()).toBeGreaterThan(0);

  const markerColors = await page.locator('.maplibregl-marker').evaluateAll(markers =>
    markers.map(marker => marker.style.backgroundColor)
  );
  expect(markerColors.some(color => color === 'rgb(34, 197, 94)' || color === '#22c55e')).toBeTruthy();
  expect(markerColors.some(color => color === 'rgb(245, 158, 11)' || color === '#f59e0b')).toBeTruthy();

  const viewer = page.getByTestId('before-after-viewer');
  await expect(viewer).toBeVisible();
  const clip = page.getByTestId('after-image-clip');
  const beforeWidth = await clip.evaluate(element => element.getBoundingClientRect().width);
  const box = await viewer.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box.x + box.width * 0.25, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  const afterWidth = await clip.evaluate(element => element.getBoundingClientRect().width);
  expect(afterWidth).toBeGreaterThan(beforeWidth);

  const reviewResponse = page.waitForResponse(response =>
    response.url().includes('/review/') && response.request().method() === 'POST'
  );
  await page.getByPlaceholder(/Enter analyst justification notes/).fill('Confirmed during Playwright E2E verification.');
  await page.getByRole('button', { name: /Confirm Change/ }).click();
  const review = await reviewResponse;
  expect(review.status()).toBe(200);
  await expect(page.getByText(/Review recorded: confirmed/i)).toBeVisible();
});