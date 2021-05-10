if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register("/serviceworker.js").then(() => {
        console.log("Service worker berhasil dipasang.");
    }).catch(function (err) {
        console.error('Tidak dapat memasang service worker.', err);
    });
}