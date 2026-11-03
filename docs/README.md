# Lobsort website

The website is a self-contained static page in `docs/index.html`. No build step or third-party scripts are needed.

## GitHub Pages

A deployment workflow is included at `.github/workflows/pages.yml`.

1. In this repository, open **Settings → Pages**.
2. Set **Source** to **GitHub Actions** (if available on the account's plan).
3. Run **Deploy documentation to GitHub Pages** from the **Actions** tab.
4. Use the URL returned by the successful deployment. Later changes to `docs/` on `main` redeploy automatically.

GitHub Pages availability for a private repository depends on the account's plan. Publication may expose the documentation site publicly; review the Pages settings before enabling it. This setup does not change repository visibility.

## Download access

The Download ZIP button points to the current main-branch archive. Because this repository is private, visitors must sign in to GitHub and have repository access. Publishing the documentation does not make the source ZIP public.

## Editing and verification

Edit `docs/index.html`. The page includes an animated spike introduction and traces, reduced-motion support, a pause control, an illustrative threshold demo, operating-system installation tabs, and a source ZIP link. Synthetic illustrations are labeled. Content reflects README.md and pyproject.toml at the time of creation.
