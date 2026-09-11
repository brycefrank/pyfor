import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

const BASE = "/pyfor";

// Old Sphinx site published every page under /html/, so keep those URLs alive.
// Astro does not apply `base` to redirect destinations, so they carry it here.
const legacyRoutes = {
  "/html/index.html": "/",
  "/html/introduction.html": "/introduction/",
  "/html/installation.html": "/installation/",
  "/html/gettingstarted.html": "/gettingstarted/",
  "/html/structure.html": "/structure/",
  "/html/topics/index.html": "/topics/canopyheightmodel/",
  "/html/topics/canopyheightmodel.html": "/topics/canopyheightmodel/",
  "/html/topics/clipping.html": "/topics/clipping/",
  "/html/topics/normalization.html": "/topics/normalization/",
  "/html/topics/metrics.html": "/topics/metrics/",
  "/html/topics/benchmarks.html": "/topics/benchmarks/",
  "/html/advanced/index.html": "/advanced/handlinglargeacquisitions/",
  "/html/advanced/handlinglargeacquisitions.html": "/advanced/handlinglargeacquisitions/",
  "/html/advanced/understandingcomponents.html": "/structure/",
  "/html/advanced/groundfilter.html": "/api/pyfor.ground_filter/",
  "/html/api/index.html": "/api/",
  "/html/api/modules.html": "/api/",
  "/html/api/pyfor.html": "/api/",
  "/html/api/pyfor.clip.html": "/api/pyfor.clip/",
  "/html/api/pyfor.cloud.html": "/api/pyfor.cloud/",
  "/html/api/pyfor.collection.html": "/api/pyfor.collection/",
  "/html/api/pyfor.gisexport.html": "/api/pyfor.gisexport/",
  "/html/api/pyfor.ground_filter.html": "/api/pyfor.ground_filter/",
  "/html/api/pyfor.metrics.html": "/api/pyfor.metrics/",
  "/html/api/pyfor.rasterizer.html": "/api/pyfor.rasterizer/",
  "/html/api/pyfor.voxelizer.html": "/api/pyfor.voxelizer/",
};

const redirects = Object.fromEntries(
  Object.entries(legacyRoutes).map(([from, to]) => [from, `${BASE}${to}`]),
);

export default defineConfig({
  site: "https://brycefrank.com",
  base: BASE,
  trailingSlash: "ignore",
  redirects,
  integrations: [
    starlight({
      title: "pyfor",
      description:
        "Python tools for processing point cloud data in large scale forest inventory systems.",
      logo: { src: "./src/assets/logo.png", alt: "pyfor" },
      favicon: "/logo.png",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/brycefrank/pyfor",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/brycefrank/pyfor/edit/master/docs/",
      },
      sidebar: [
        { label: "Introduction", link: "/introduction/" },
        { label: "Installation", link: "/installation/" },
        { label: "Getting Started", link: "/gettingstarted/" },
        {
          label: "The Basics",
          items: [
            { label: "Canopy Height Models", link: "/topics/canopyheightmodel/" },
            { label: "Clipping", link: "/topics/clipping/" },
            { label: "Normalization", link: "/topics/normalization/" },
            { label: "Area-Based Metrics", link: "/topics/metrics/" },
            { label: "Benchmarks", link: "/topics/benchmarks/" },
          ],
        },
        {
          label: "Advanced",
          items: [
            { label: "The Structure of pyfor", link: "/structure/" },
            {
              label: "Handling Large Acquisitions",
              link: "/advanced/handlinglargeacquisitions/",
            },
          ],
        },
        {
          label: "API Reference",
          items: [{ autogenerate: { directory: "api" } }],
        },
      ],
    }),
  ],
});
