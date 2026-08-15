import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://qtensor.com.au',
  base: '/',
  integrations: [tailwind()],
  output: 'static',
});
