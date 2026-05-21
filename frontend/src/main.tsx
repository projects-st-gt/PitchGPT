import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { StrikeZonePreview } from "./StrikeZonePreview";
import "./index.css";

// Migration-time helper: ?preview=zone renders just the StrikeZone widget
// with mock data for visual verification. Remove together with
// StrikeZonePreview.tsx when the 14-zone migration is signed off.
const params = new URLSearchParams(window.location.search);
const RootComponent = params.get("preview") === "zone" ? StrikeZonePreview : App;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RootComponent />
  </React.StrictMode>,
);
