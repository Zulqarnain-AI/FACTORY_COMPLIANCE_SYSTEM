import { useState, useEffect, useRef, useCallback } from "react";
import {
  AlertTriangle, Shield, Activity, Upload, Download,
  Wifi, WifiOff, Bell, BellOff, ChevronRight, X,
  Filter, RefreshCw, Eye, AlertOctagon, Info, CheckCircle
} from "lucide-react";

const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/alerts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SEVERITY_CONFIG = {
  LOW:      { color: "text-emerald-400", bg: "bg-emerald-900/40", border: "border-emerald-500", badge: "bg-emerald-800 text-emerald-200", dot: "bg-emerald-400" },
  MEDIUM:   { color: "text-yellow-400",  bg: "bg-yellow-900/40",  border: "border-yellow-500",  badge: "bg-yellow-800 text-yellow-200",  dot: "bg-yellow-400" },
  HIGH:     { color: "text-orange-400",  bg: "bg-orange-900/40",  border: "border-orange-500",  badge: "bg-orange-800 text-orange-200",  dot: "bg-orange-400" },
  CRITICAL: { color: "text-red-400",     bg: "bg-red-900/40",     border: "border-red-500",     badge: "bg-red-800 text-red-200",     dot: "bg-red-400" },
};

const CLASS_ICONS = {
  "Safe Walkway Violation":        "🚶",
  "Unauthorized Intervention":     "🦺",
  "Opened Panel Cover":            "⚡",
  "Carrying Overload with Forklift": "🏗️",
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

const fmt = (ts) => ts ? new Date(ts).toLocaleString() : "—";
const shortId = (id) => id ? id.slice(0, 8).toUpperCase() : "—";

function SeverityBadge({ severity }) {
  const cfg = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.LOW;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold ${cfg.badge}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {severity}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Toast Notification (HIGH / CRITICAL real-time alerts)
// ---------------------------------------------------------------------------

function Toast({ alert, onDismiss }) {
  const isCritical = alert.severity === "CRITICAL";
  const cfg = SEVERITY_CONFIG[alert.severity];

  useEffect(() => {
    const timer = setTimeout(onDismiss, isCritical ? 12000 : 8000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`toast-enter relative flex gap-3 p-4 rounded-lg border ${cfg.bg} ${cfg.border} 
      ${isCritical ? "critical-pulse" : ""} max-w-sm shadow-2xl`}>
      <div className="shrink-0 mt-0.5">
        {isCritical
          ? <AlertOctagon className="w-5 h-5 text-red-400" />
          : <AlertTriangle className="w-5 h-5 text-orange-400" />
        }
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <SeverityBadge severity={alert.severity} />
          <span className="text-xs text-gray-400">{alert.zone}</span>
        </div>
        <p className={`text-sm font-semibold ${cfg.color}`}>
          {CLASS_ICONS[alert.behavior_class]} {alert.behavior_class}
        </p>
        <p className="text-xs text-gray-300 mt-1 line-clamp-2">{alert.event_description}</p>
        <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
          <span>{shortId(alert.event_id)}</span>
          <span>·</span>
          <span>{alert.policy_rule_ref}</span>
          <span>·</span>
          <span>{fmt(alert.timestamp)}</span>
        </div>
      </div>
      <button onClick={onDismiss} className="shrink-0 text-gray-500 hover:text-gray-300">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

function ToastContainer({ alerts, onDismiss }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-3 max-h-screen overflow-y-auto pointer-events-none">
      {alerts.map(a => (
        <div key={a.event_id} className="pointer-events-auto">
          <Toast alert={a} onDismiss={() => onDismiss(a.event_id)} />
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// View A: Live Feed Monitor
// ---------------------------------------------------------------------------

function LiveFeedView({ stats, wsConnected }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [activeSnapshot, setActiveSnapshot] = useState(null);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
    setActiveSnapshot(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/api/process`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
      // Show first snapshot if available
      const snap = data.violations?.find(v => v.frame_snapshot_b64);
      if (snap) setActiveSnapshot(snap.frame_snapshot_b64);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Upload Panel */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h2 className="text-sm font-bold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2">
          <Upload className="w-4 h-4 text-blue-400" /> Upload Video Clip
        </h2>

        <label className={`flex flex-col items-center justify-center gap-3 p-8 rounded-lg border-2 border-dashed 
          cursor-pointer transition-colors ${file ? "border-blue-500 bg-blue-900/10" : "border-gray-700 hover:border-gray-500"}`}>
          <Upload className="w-8 h-8 text-gray-500" />
          <div className="text-center">
            <p className="text-sm text-gray-300">{file ? file.name : "Drop video clip here"}</p>
            <p className="text-xs text-gray-500 mt-1">MP4, AVI, MOV · 3–18s clips</p>
          </div>
          <input type="file" accept="video/*" onChange={handleFileChange} className="hidden" />
        </label>

        {previewUrl && (
          <video src={previewUrl} controls className="w-full rounded-lg mt-4 border border-gray-700" />
        )}

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="w-full mt-4 py-3 px-4 rounded-lg font-bold text-sm bg-blue-600 hover:bg-blue-500 
            disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed transition-colors"
        >
          {uploading ? (
            <span className="flex items-center justify-center gap-2">
              <RefreshCw className="w-4 h-4 animate-spin" /> Analyzing Compliance...
            </span>
          ) : "Run Compliance Detection"}
        </button>

        {error && (
          <div className="mt-4 p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm">
            ⚠️ {error}
          </div>
        )}

        {/* WS connection status */}
        <div className={`mt-4 flex items-center gap-2 text-xs ${wsConnected ? "text-emerald-400" : "text-red-400"}`}>
          {wsConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
          {wsConnected ? "Alert stream connected" : "Alert stream disconnected"}
        </div>
      </div>

      {/* Detection Results */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h2 className="text-sm font-bold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2">
          <Eye className="w-4 h-4 text-purple-400" /> Detection Results
        </h2>

        {!result && !uploading && (
          <div className="flex flex-col items-center justify-center h-48 text-gray-600">
            <Shield className="w-10 h-10 mb-3" />
            <p className="text-sm">Upload a clip to see violations</p>
          </div>
        )}

        {result && (
          <div>
            {/* Summary bar */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              {Object.entries(SEVERITY_CONFIG).map(([sev, cfg]) => {
                const count = result.severity_breakdown?.[sev] || 0;
                return (
                  <div key={sev} className={`p-3 rounded-lg ${cfg.bg} border ${cfg.border}`}>
                    <div className={`text-2xl font-black ${cfg.color}`}>{count}</div>
                    <div className={`text-xs font-semibold ${cfg.color}`}>{sev}</div>
                  </div>
                );
              })}
            </div>

            {/* Annotated frame snapshot */}
            {activeSnapshot && (
              <div className="mb-4">
                <p className="text-xs text-gray-500 mb-2">ANNOTATED FRAME</p>
                <img
                  src={`data:image/jpeg;base64,${activeSnapshot}`}
                  alt="Violation frame"
                  className="w-full rounded border border-gray-700"
                />
              </div>
            )}

            {/* Violation list */}
            {result.violations?.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {result.violations.map((v) => {
                  const cfg = SEVERITY_CONFIG[v.severity] || SEVERITY_CONFIG.LOW;
                  return (
                    <div
                      key={v.event_id}
                      onClick={() => v.frame_snapshot_b64 && setActiveSnapshot(v.frame_snapshot_b64)}
                      className={`p-3 rounded-lg border ${cfg.bg} ${cfg.border} cursor-pointer hover:opacity-80 transition-opacity`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm">{CLASS_ICONS[v.behavior_class]} {v.behavior_class}</span>
                        <SeverityBadge severity={v.severity} />
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        {v.policy_rule_ref} · Frame {v.frame_number} · {Math.round(v.confidence * 100)}% conf
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-emerald-400 text-sm">
                <CheckCircle className="w-4 h-4" /> No violations detected
              </div>
            )}
          </div>
        )}
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="lg:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(stats.by_severity || {}).map(([sev, count]) => {
            const cfg = SEVERITY_CONFIG[sev] || SEVERITY_CONFIG.LOW;
            return (
              <div key={sev} className={`bg-gray-900 rounded-xl border ${cfg.border} p-4`}>
                <div className={`text-3xl font-black ${cfg.color}`}>{count}</div>
                <div className="text-xs text-gray-400 font-semibold mt-1">{sev} VIOLATIONS</div>
                <div className="text-xs text-gray-600 mt-1">all time</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// View B: Alert Timeline Stream
// ---------------------------------------------------------------------------

function AlertTimelineView({ timeline }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [timeline]);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 h-[600px] flex flex-col">
      <h2 className="text-sm font-bold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2 shrink-0">
        <Activity className="w-4 h-4 text-yellow-400" /> Real-time Alert Timeline
        <span className="ml-auto text-xs text-gray-500 normal-case font-normal">
          {timeline.length} events
        </span>
      </h2>

      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {timeline.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <Bell className="w-10 h-10 mb-3" />
            <p className="text-sm">No alerts yet — process a clip to begin</p>
          </div>
        )}

        {timeline.map((event, i) => {
          const cfg = SEVERITY_CONFIG[event.severity] || SEVERITY_CONFIG.LOW;
          return (
            <div key={event.event_id || i}
              className={`relative flex gap-4 p-4 rounded-lg border ${cfg.bg} ${cfg.border}`}>
              {/* Timeline line */}
              {i < timeline.length - 1 && (
                <div className="absolute left-7 top-12 bottom-0 w-px bg-gray-800" />
              )}
              {/* Icon */}
              <div className={`shrink-0 w-6 h-6 rounded-full flex items-center justify-center z-10
                ${event.severity === "CRITICAL" ? "bg-red-600" : "bg-orange-600"}`}>
                <span className="text-xs">
                  {event.severity === "CRITICAL" ? "!" : "↑"}
                </span>
              </div>
              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <SeverityBadge severity={event.severity} />
                  <span className="text-sm font-semibold text-white">
                    {CLASS_ICONS[event.behavior_class]} {event.behavior_class}
                  </span>
                  <span className="text-xs text-gray-400">{event.zone}</span>
                </div>
                <p className="text-xs text-gray-300 mt-1 line-clamp-2">{event.event_description}</p>
                <div className="flex flex-wrap gap-3 mt-2 text-xs text-gray-500">
                  <span>{shortId(event.event_id)}</span>
                  <span>·</span>
                  <span>{event.policy_rule_ref}</span>
                  <span>·</span>
                  <span>{event.clip_id}</span>
                  <span>·</span>
                  <span>{fmt(event.timestamp)}</span>
                </div>
              </div>
              <ChevronRight className="shrink-0 w-4 h-4 text-gray-600 self-center" />
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// View C: Historical Log & Export
// ---------------------------------------------------------------------------

function HistoricalLogView() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({ severity: "", behavior_class: "" });
  const [error, setError] = useState(null);

  const fetchReports = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (filters.severity) params.set("severity", filters.severity);
      if (filters.behavior_class) params.set("behavior_class", filters.behavior_class);
      const res = await fetch(`${API}/api/reports?${params}&limit=200`);
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setReports(data.reports || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => { fetchReports(); }, []);

  const handleExport = async (format) => {
    const params = new URLSearchParams();
    if (filters.severity) params.set("severity", filters.severity);
    const res = await fetch(`${API}/api/export/${format}?${params}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `compliance_log.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const CLASSES = [
    "", "Safe Walkway Violation", "Unauthorized Intervention",
    "Opened Panel Cover", "Carrying Overload with Forklift"
  ];

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
      <div className="flex items-center justify-between flex-wrap gap-4 mb-6">
        <h2 className="text-sm font-bold text-gray-300 uppercase tracking-widest flex items-center gap-2">
          <Shield className="w-4 h-4 text-blue-400" /> Compliance Log
          <span className="text-gray-500 font-normal normal-case">({reports.length} records)</span>
        </h2>
        <div className="flex gap-2">
          <button onClick={() => handleExport("csv")}
            className="flex items-center gap-1 px-3 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors">
            <Download className="w-3 h-3" /> CSV
          </button>
          <button onClick={() => handleExport("json")}
            className="flex items-center gap-1 px-3 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors">
            <Download className="w-3 h-3" /> JSON
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-6">
        <div className="flex items-center gap-2 bg-gray-800 rounded px-3 py-2">
          <Filter className="w-3 h-3 text-gray-400" />
          <select
            value={filters.severity}
            onChange={e => setFilters(f => ({ ...f, severity: e.target.value }))}
            className="text-xs bg-transparent text-gray-200 outline-none"
          >
            <option value="">All Severities</option>
            {["LOW", "MEDIUM", "HIGH", "CRITICAL"].map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2 bg-gray-800 rounded px-3 py-2">
          <Filter className="w-3 h-3 text-gray-400" />
          <select
            value={filters.behavior_class}
            onChange={e => setFilters(f => ({ ...f, behavior_class: e.target.value }))}
            className="text-xs bg-transparent text-gray-200 outline-none"
          >
            {CLASSES.map(c => <option key={c} value={c}>{c || "All Behavior Classes"}</option>)}
          </select>
        </div>

        <button onClick={fetchReports}
          className="flex items-center gap-1 px-3 py-2 text-xs rounded bg-blue-700 hover:bg-blue-600 text-white transition-colors">
          <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} /> Refresh
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800">
              {["Event ID", "Timestamp", "Clip", "Zone", "Behavior Class", "Policy Ref", "Severity", "Action"].map(h => (
                <th key={h} className="text-left py-2 px-3 text-gray-500 font-semibold uppercase tracking-wider whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {reports.length === 0 && !loading && (
              <tr>
                <td colSpan={8} className="text-center py-12 text-gray-600">
                  No compliance records found. Process a video clip to generate reports.
                </td>
              </tr>
            )}
            {reports.map((r) => {
              const cfg = SEVERITY_CONFIG[r.severity] || SEVERITY_CONFIG.LOW;
              return (
                <tr key={r.event_id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                  <td className="py-2 px-3 font-mono text-gray-400">{shortId(r.event_id)}</td>
                  <td className="py-2 px-3 text-gray-400 whitespace-nowrap">{fmt(r.timestamp)}</td>
                  <td className="py-2 px-3 text-gray-300 max-w-24 truncate" title={r.clip_id}>{r.clip_id}</td>
                  <td className="py-2 px-3 text-gray-400">{r.zone}</td>
                  <td className="py-2 px-3 text-gray-200 whitespace-nowrap">
                    {CLASS_ICONS[r.behavior_class]} {r.behavior_class}
                  </td>
                  <td className="py-2 px-3 text-blue-400">{r.policy_rule_ref}</td>
                  <td className="py-2 px-3"><SeverityBadge severity={r.severity} /></td>
                  <td className={`py-2 px-3 ${cfg.color} text-xs`}>
                    {r.escalation_action === "Real-time alert triggered + DB log"
                      ? "🔔 Alert + Log"
                      : "📋 Log only"
                    }
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  const [activeView, setActiveView] = useState("live");
  const [stats, setStats] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [alertTimeline, setAlertTimeline] = useState([]);
  const [toasts, setToasts] = useState([]);
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  // --- Fetch dashboard stats ---
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/stats`);
      if (res.ok) setStats(await res.json());
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  // --- WebSocket connection ---
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      clearTimeout(reconnectRef.current);
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "VIOLATION_ALERT") {
        // Add to timeline (newest first for display)
        setAlertTimeline(prev => [...prev.slice(-99), msg]);

        // Show toast for HIGH/CRITICAL
        if (["HIGH", "CRITICAL"].includes(msg.severity)) {
          setToasts(prev => [...prev.slice(-4), msg]);
          fetchStats();  // Refresh stats on new alert
        }
      }
    };

    ws.onclose = () => {
      setWsConnected(false);
      reconnectRef.current = setTimeout(connectWs, 3000);
    };

    ws.onerror = () => ws.close();
  }, [fetchStats]);

  useEffect(() => {
    connectWs();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connectWs]);

  const dismissToast = (id) => setToasts(prev => prev.filter(t => t.event_id !== id));

  const VIEWS = [
    { id: "live",     label: "Live Monitor",    icon: Eye },
    { id: "timeline", label: "Alert Timeline",  icon: Activity },
    { id: "log",      label: "Compliance Log",  icon: Shield },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Real-time toast notifications */}
      <ToastContainer alerts={toasts} onDismiss={dismissToast} />

      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shrink-0">
                <Shield className="w-5 h-5" />
              </div>
              <div>
                <h1 className="font-black text-base tracking-tight leading-none">
                  FACTORY COMPLIANCE SYSTEM
                </h1>
                <p className="text-xs text-gray-500 mt-0.5 tracking-widest">
                  KMP-OHS-POL-001 · KAFAOGLU METAL PLASTİK
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* WebSocket status */}
              <div className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border
                ${wsConnected ? "border-emerald-700 bg-emerald-900/30 text-emerald-400" : "border-red-700 bg-red-900/30 text-red-400"}`}>
                {wsConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                {wsConnected ? "LIVE" : "OFFLINE"}
              </div>

              {/* Toast count indicator */}
              {toasts.length > 0 && (
                <div className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full 
                  border border-red-700 bg-red-900/30 text-red-400 animate-pulse">
                  <Bell className="w-3 h-3" />
                  {toasts.length} active
                </div>
              )}

              {/* Total violations */}
              {stats && (
                <div className="text-xs text-gray-400">
                  <span className="text-white font-bold">{stats.total_violations}</span> total violations
                </div>
              )}
            </div>
          </div>

          {/* Navigation tabs */}
          <nav className="flex gap-1 mt-4">
            {VIEWS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveView(id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all
                  ${activeView === id
                    ? "bg-blue-600 text-white"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"}`}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
                {id === "timeline" && alertTimeline.length > 0 && (
                  <span className="bg-red-600 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                    {Math.min(alertTimeline.length, 9)}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Policy quick reference banner */}
      <div className="bg-blue-950/30 border-b border-blue-900/40">
        <div className="max-w-7xl mx-auto px-6 py-2 flex flex-wrap gap-4 text-xs text-blue-300">
          <span className="flex items-center gap-1"><Info className="w-3 h-3" /> Policy classes:</span>
          {["🚶 Walkway (§3.3.2) → HIGH", "🦺 Intervention (§4.3.2) → CRITICAL",
            "⚡ Panel (§5.2.2) → HIGH", "🏗️ Forklift (§6.3.2) → CRITICAL"].map(t => (
            <span key={t} className="text-blue-400 font-medium">{t}</span>
          ))}
        </div>
      </div>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeView === "live"     && <LiveFeedView stats={stats} wsConnected={wsConnected} />}
        {activeView === "timeline" && <AlertTimelineView timeline={alertTimeline} />}
        {activeView === "log"      && <HistoricalLogView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between flex-wrap gap-2 text-xs text-gray-600">
          <span>KMP-OHS-POL-001 · Effective 01 November 2022 · EHS Department</span>
          <span>CONTROLLED DOCUMENT — Internal Use Only</span>
        </div>
      </footer>
    </div>
  );
}
