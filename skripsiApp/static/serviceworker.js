var staticCacheName = 'pendeteksirupiahpwa-v1';

self.addEventListener('install', function (event) {
	console.log('[ServiceWorker] Install');
	event.waitUntil(
		caches.open(staticCacheName).then(function (cache) {
			console.log('[ServiceWorker] Pre-caching offline page');
			return cache.addAll([
				'/'
				// '/static/css/style.css',
				// '/static/js/main.js',
				// '/static/images/favicon-16x16.png',
				// '/static/images/favicon-32x32.png',
				// '/static/images/icon-192x192.png',
				// '/static/manifest.json'
			]);
		})
	);
});

self.addEventListener('activate', (evt) => {
	console.log('[ServiceWorker] Activate');
	evt.waitUntil(
		caches.keys().then((keyList) => {
			return Promise.all(keyList.map((key) => {
				if (key !== staticCacheName) {
					console.log('[ServiceWorker] Removing old cache', key);
					return caches.delete(key);
				}
			}));
		})
	);
	self.clients.claim();
});

self.addEventListener('fetch', function (event) {
	var requestUrl = new URL(event.request.url);

	if (requestUrl.origin === location.origin) {
		if ((requestUrl.pathname === '/')) {
			event.respondWith(caches.match('/'));
			return;
		}
	}
	// event.respondWith(
	// 	caches.match(event.request).then(function () {
	// 		return caches.match(event.request);
	// 	})
	// );
});