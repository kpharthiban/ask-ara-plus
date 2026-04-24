import type { Metadata, Viewport } from "next";
import { Plus_Jakarta_Sans, Noto_Sans } from "next/font/google";
import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

const plusJakarta = Plus_Jakarta_Sans({ 
  subsets: ["latin"],
  variable: "--font-plus-jakarta",
});

const notoSans = Noto_Sans({
  subsets: ["latin"],
  variable: "--font-noto-sans",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "AskAra+ — ASEAN Government Services Assistant",
  description:
    "Multilingual AI assistant helping ASEAN citizens navigate government services in their own language and dialect.",
  manifest: "/manifest.json",
  applicationName: "AskAra+",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "AskAra+",
  },
  formatDetection: {
    telephone: true,
  },
  openGraph: {
    type: "website",
    title: "AskAra+ — ASEAN Government Services Assistant",
    description:
      "Multilingual AI assistant helping ASEAN citizens navigate government services in their own language and dialect.",
    siteName: "AskAra+",
  },
  icons: {
    icon: [
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
    ],
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#d97706" },
    { media: "(prefers-color-scheme: dark)", color: "#0f172a" },
  ],
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* PWA: register service worker */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', () => {
                  navigator.serviceWorker.register('/sw.js')
                    .then(reg => console.log('[SW] Registered:', reg.scope))
                    .catch(err => console.warn('[SW] Registration failed:', err));
                });
              }
            `,
          }}
        />
      </head>
      <body className={`${plusJakarta.variable} ${notoSans.variable} font-sans antialiased text-slate-900 bg-slate-50 dark:bg-slate-950 dark:text-slate-100 transition-colors duration-500`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}