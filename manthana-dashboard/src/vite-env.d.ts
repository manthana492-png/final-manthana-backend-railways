/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DASHBOARD_API_BASE: string;
  readonly VITE_BIOVIL_API_BASE: string;
  readonly VITE_ORTHANC_BASE: string;
  readonly VITE_GATEWAY_PUBLIC_BASE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "cornerstone-core";
declare module "cornerstone-wado-image-loader";
