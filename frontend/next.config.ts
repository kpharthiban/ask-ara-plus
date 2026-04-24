import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Enforce standalone output for Docker optimization */
  output: "standalone",
  
  /* You can add other config options here if needed */
};

export default nextConfig;