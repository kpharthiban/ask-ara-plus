/**
 * AskAra+ Service Worker
 * 
 * Strategy:
 *   - App shell (HTML, CSS, JS, fonts) → Cache First
 *   - API calls → Network First (no caching for chat data)
 *   - Icons/images → Cache First with network fallback
 *   - Offline fallback → cached shell still renders
 */

const CACHE_NAME = "askara-v1";

// App shell files to pre-cache on install
const PRECACHE_URLS = [
  "/",
  "/manifest.json",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
];

// ── Install — pre-cache app shell ──
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(PRECACHE_URLS);
    })
  );
  // Activate immediately
  self.skipWaiting();
});

// ── Activate — clean up old caches ──
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  // Take control of all clients immediately
  self.clients.claim();
});

// ── Fetch — routing strategy ──
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests (POST, WebSocket upgrades, etc.)
  if (request.method !== "GET") return;

  // Skip API calls and WebSocket — always go to network
  if (
    url.pathname.startsWith("/api/") ||
    url.pathname.startsWith("/ws/") ||
    url.pathname.startsWith("/mcp/") ||
    url.pathname.startsWith("/chat") ||
    url.pathname.startsWith("/health")
  ) {
    return;
  }

  // Skip external requests (CDNs, analytics, etc.)
  if (url.origin !== self.location.origin) return;

  // App shell + static assets → Cache First, fallback to network
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;

      return fetch(request)
        .then((response) => {
          // Don't cache non-ok responses or opaque responses
          if (!response || response.status !== 200 || response.type !== "basic") {
            return response;
          }

          // Cache a clone for next time
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(request, responseToCache);
          });

          return response;
        })
        .catch(() => {
          // Offline fallback: for navigation requests, return cached root
          if (request.mode === "navigate") {
            return caches.match("/");
          }
          return new Response("Offline", { status: 503 });
        });
    })
  );
});