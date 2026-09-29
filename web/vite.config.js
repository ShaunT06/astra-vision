import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The backend URL is read from REACT_APP_BACKEND_URL (build-time). Empty = same-origin /api.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "REACT_APP_");
  return {
    plugins: [react()],
    define: { "process.env.REACT_APP_BACKEND_URL": JSON.stringify(env.REACT_APP_BACKEND_URL || "") },
    server: { proxy: { "/api": "http://localhost:8000" } },
  };
});
