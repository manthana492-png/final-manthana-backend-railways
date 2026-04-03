import type { Metadata } from "next";
import "./globals.css";
import ThemeProvider from "@/components/shared/ThemeProvider";

export const metadata: Metadata = {
  title: "Manthana Radiologist Copilot",
  description:
    "AI-powered radiology second-opinion suite — 13 specialized services, 23+ models, for clinical decision support across X-Ray, CT, MRI, Ultrasound, ECG, Pathology, and more.",
  keywords: [
    "radiology", "AI", "medical imaging", "X-ray", "CT scan", "MRI",
    "pathology", "ECG", "India", "clinical decision support",
  ],
  openGraph: {
    title: "Manthana Radiologist Copilot",
    description: "India's AI radiology second-opinion suite",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        {/* Anti-flicker: apply theme before first paint */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("manthana_theme");if(t&&["default","blackhole","clinical"].indexOf(t)!==-1){if(t==="clinical"&&window.innerWidth<=1024)t="default";document.documentElement.dataset.theme=t}}catch(e){}})()`,
          }}
        />
      </head>
      <body className="cosmic-bg">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
