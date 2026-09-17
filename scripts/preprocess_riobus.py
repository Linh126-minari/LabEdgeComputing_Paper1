"""
Rio Bus GPS Preprocessing Pipeline (Phase 4 - T4.1, bản dữ liệu thật)
=====================================================================
Nguồn: GPS xe buýt Rio de Janeiro (SMTR/data.rio), 25/01–21/03/2019,
`data/unstructured/individual_bus_lines_data/bus_line_*/part-*.csv`
(cột: date, time, order, line, latitude, longitude, speed[km/h]).

Các bước (chi tiết + số liệu: BAO_CAO_PHASE4_RIOBUS.md):
  1. clean_raw        — sửa timestamp (cột `time` mang ngày giả 2025-11-25 của Spark;
                        ngày thật ở cột `date`), ép kiểu, lọc bbox, bỏ trùng, lọc nhảy GPS,
                        bỏ ngày lễ, tách chuyến theo khoảng trống.
  2. find_hotspot     — chọn vùng 1 km² có nhiều xe buýt nhất giờ cao điểm; chiếu
                        lat/lon → mét (equirectangular cục bộ, sai số < 0.1% trong vài km).
  3. extract_window   — với mỗi (ngày, giờ bắt đầu): các đoạn quỹ đạo thật trên cùng một
                        đồng hồ → quan hệ không gian giữa các xe là thật.
                        Nội suy tuyến tính ở `out_dt` (không nội suy qua khoảng trống lớn);
                        xe đi vào/ra được gán nối tiếp vào các slot cố định.
  4. split theo ngày  — train/test không chồng lấn thời gian.
  5. stats            — thống kê dùng cho bài (vận tốc, lưu trú MEC, mật độ láng giềng).

HẠN CHẾ (ghi trong bài): GPS lấy mẫu trung vị ~70 s → nội suy thẳng giữa hai điểm, chưa
map-matching theo bản đồ đường; chỉ có xe buýt (thiếu xe con).
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import Logger

logger = Logger("RioBusPreprocess")

EARTH_R = 6371000.0
OUTPUT_COLUMNS = ["window_id", "vehicle_id", "bus_id", "line", "t_rel", "x_m", "y_m", "v_mps", "active"]


# ============================================================ 1. Load & clean

def load_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("bus_line_*/part-*.csv"))
    if not files:
        raise FileNotFoundError(f"Không thấy part-*.csv trong {raw_dir}")
    logger.info(f"Đọc {len(files)} file từ {raw_dir} ...")
    return pd.concat((pd.read_csv(f, dtype=str) for f in files), ignore_index=True)


def clean_raw(df: pd.DataFrame, cfg: dict):
    """Trả về (df sạch, dict số dòng bị loại ở từng bước)."""
    report = {"raw_rows": int(len(df))}

    out = pd.DataFrame({
        "order": df["order"].astype(str),
        "line": df["line"].astype(str).str.replace(r"\.0$", "", regex=True),
        "lat": pd.to_numeric(df["latitude"], errors="coerce"),
        "lon": pd.to_numeric(df["longitude"], errors="coerce"),
        "speed_kmh": pd.to_numeric(df["speed"], errors="coerce"),
        # `time` = "2025-11-25T08:57:38.000Z" (ngày giả) → chỉ lấy HH:MM:SS, ghép với `date`
        "ts": pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str).str[11:19],
                             format="%m-%d-%Y %H:%M:%S", errors="coerce"),
    })
    n = len(out)
    out = out.dropna()
    report["dropped_unparseable"] = n - len(out)

    b = cfg["bbox"]
    n = len(out)
    out = out[out.lat.between(b["lat_min"], b["lat_max"]) & out.lon.between(b["lon_min"], b["lon_max"])]
    report["dropped_outside_bbox"] = n - len(out)

    n = len(out)
    excl = pd.to_datetime(pd.Series(cfg.get("exclude_dates", []), dtype=str)).dt.date.tolist()
    out = out[~out.ts.dt.date.isin(excl)]
    report["dropped_excluded_dates"] = n - len(out)

    n = len(out)
    out = out[(out.speed_kmh >= 0) & (out.speed_kmh <= cfg["max_recorded_speed_kmh"])]
    report["dropped_bad_recorded_speed"] = n - len(out)

    n = len(out)
    out = out.sort_values(["order", "ts"]).drop_duplicates(["order", "ts"]).reset_index(drop=True)
    report["dropped_duplicates"] = n - len(out)

    # Nhảy GPS: (a) bỏ điểm "gai" — vận tốc vào VÀ ra đều > ngưỡng (giữ điểm tốt kế sau);
    # (b) lặp bỏ điểm có vận tốc vào > ngưỡng còn sót (bỏ 1 điểm làm đổi cặp kế).
    n = len(out)
    th = cfg["max_implied_speed_kmh"]
    for _ in range(3):
        v_in = _implied_speed_kmh(out)
        v_out = v_in.groupby(out.order).shift(-1).fillna(0.0)
        spike = (v_in > th) & (v_out > th)
        if not spike.any():
            break
        out = out[~spike].reset_index(drop=True)
    for _ in range(5):
        bad = _implied_speed_kmh(out) > th
        if not bad.any():
            break
        out = out[~bad].reset_index(drop=True)
    report["dropped_gps_jumps"] = n - len(out)

    # Tách chuyến
    gap = out.groupby("order").ts.diff().dt.total_seconds()
    new_trip = gap.isna() | (gap > cfg["max_trip_gap_s"])
    out["trip_id"] = new_trip.cumsum().astype(np.int64)
    n = len(out)
    sizes = out.groupby("trip_id").trip_id.transform("size")
    out = out[sizes >= cfg["min_trip_points"]].reset_index(drop=True)
    report["dropped_short_trips"] = n - len(out)

    report["clean_rows"] = int(len(out))
    report["buses"] = int(out.order.nunique())
    report["lines"] = int(out.line.nunique())
    report["trips"] = int(out.trip_id.nunique())
    report = {k: int(v) for k, v in report.items()}
    return out, report


def _implied_speed_kmh(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("order")
    dt = g.ts.diff().dt.total_seconds()
    dist = haversine_m(g.lat.shift(), g.lon.shift(), df.lat, df.lon)
    v = dist / dt * 3.6
    return v.where(dt > 0, 0.0).fillna(0.0)


def haversine_m(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


# ============================================================ 2. Projection & hotspot

def project_local(lat, lon, lat0: float, lon0: float, anchor_xy=(0.0, 0.0)):
    """Equirectangular quanh (lat0, lon0) ↦ anchor_xy. x = Đông, y = Bắc (mét)."""
    x = np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * EARTH_R + anchor_xy[0]
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R + anchor_xy[1]
    return x, y


def find_hotspot(df: pd.DataFrame, cfg: dict):
    if cfg.get("hotspot_center"):
        return tuple(cfg["hotspot_center"])
    peak = df[df.ts.dt.hour.isin(cfg["hotspot_hours"]) & (df.ts.dt.dayofweek < 5)]
    lat0, lon0 = peak.lat.mean(), peak.lon.mean()
    x, y = project_local(peak.lat, peak.lon, lat0, lon0)
    cell = cfg["hotspot_grid_m"]
    key = np.floor(x / cell).astype(np.int64) * 100000 + np.floor(y / cell).astype(np.int64)
    # Đếm cặp (xe, giờ) khác nhau thay vì số điểm → không thiên vị xe dừng lâu
    visits = pd.DataFrame({"cell": key, "order": peak.order.to_numpy(), "h": peak.ts.dt.floor("h").to_numpy()})
    best = visits.drop_duplicates().cell.value_counts().idxmax()
    in_cell = key == best
    return float(peak.lat.to_numpy()[in_cell].mean()), float(peak.lon.to_numpy()[in_cell].mean())


# ============================================================ 3. Windows

def _window_segments(df_day: pd.DataFrame, t0: pd.Timestamp, center, cfg: dict, t_grid: np.ndarray):
    """Các đoạn quỹ đạo thật trong [t0, t0+D]: không có khoảng trống > max_interp_gap_s,
    cắt trong clip_radius_m, dài ≥ min_segment_s, và đi vào trong keep_radius_m quanh hotspot."""
    D, gap_max = cfg["window_duration_s"], cfg["max_interp_gap_s"]
    ax, ay = cfg["anchor_xy"]
    sub = df_day[(df_day.ts >= t0 - pd.Timedelta(seconds=gap_max)) &
                 (df_day.ts <= t0 + pd.Timedelta(seconds=D + gap_max))]
    segs = []
    for trip_id, tr in sub.groupby("trip_id", sort=False):
        t = (tr.ts - t0).dt.total_seconds().to_numpy()
        x, y = project_local(tr.lat.to_numpy(), tr.lon.to_numpy(), center[0], center[1], cfg["anchor_xy"])
        cuts = np.where(np.diff(t) > gap_max)[0] + 1
        for idx in np.split(np.arange(len(t)), cuts):
            if len(idx) < 2:
                continue
            ts_ = t[idx]
            g = (t_grid >= ts_[0]) & (t_grid <= ts_[-1])
            if not g.any():
                continue
            gi = np.where(g)[0]
            xi, yi = np.interp(t_grid[gi], ts_, x[idx]), np.interp(t_grid[gi], ts_, y[idx])
            # chỉ giữ phần quỹ đạo trong clip_radius_m; mỗi lượt vào vùng là một đoạn riêng
            inside = np.hypot(xi - ax, yi - ay) <= cfg["clip_radius_m"]
            edges = np.diff(np.r_[0, inside.astype(int), 0])
            for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                if (b - a) * cfg["out_dt"] < cfg["min_segment_s"]:
                    continue
                d_min = np.hypot(xi[a:b] - ax, yi[a:b] - ay).min()
                if d_min > cfg["keep_radius_m"]:
                    continue
                segs.append({"bus": str(tr.order.iloc[0]), "line": str(tr.line.iloc[0]),
                             "i0": int(gi[a]), "i1": int(gi[b - 1] + 1),
                             "x": xi[a:b], "y": yi[a:b], "d_min": d_min})
    return segs


def extract_window(df_day: pd.DataFrame, t0: pd.Timestamp, center, cfg: dict, window_id: str) -> pd.DataFrame:
    """Gán các đoạn quỹ đạo thật vào `num_slots` slot (lập lịch khoảng, best-fit).

    Mỗi slot = một thực thể của simulator; khi xe buýt rời vùng/mất tín hiệu, xe buýt
    thật kế tiếp nhận slot. Mọi slot dùng chung đồng hồ t_rel → khoảng cách V2V là thật.
    Khoảng không có xe: active=0, giữ vị trí gần nhất, v=0.
    """
    t_grid = np.arange(0.0, cfg["window_duration_s"] + 1e-9, cfg["out_dt"])
    n, K = len(t_grid), cfg["num_slots"]
    segs = sorted(_window_segments(df_day, t0, center, cfg, t_grid), key=lambda s: (s["i0"], s["d_min"]))
    slot_end = np.full(K, -1)
    X, Y = np.full((K, n), np.nan), np.full((K, n), np.nan)
    bus = np.full((K, n), "", dtype=object); line = np.full((K, n), "", dtype=object)
    for s in segs:
        free = np.where(slot_end <= s["i0"])[0]
        if free.size == 0:
            continue
        k = free[np.argmax(slot_end[free])]          # best-fit: slot rảnh gần nhất → ít lỗ
        X[k, s["i0"]:s["i1"]], Y[k, s["i0"]:s["i1"]] = s["x"], s["y"]
        bus[k, s["i0"]:s["i1"]], line[k, s["i0"]:s["i1"]] = s["bus"], s["line"]
        slot_end[k] = s["i1"]

    active = ~np.isnan(X)
    coverage = active.mean(axis=1)
    if (coverage > 0).sum() < K or coverage.mean() < cfg["min_window_coverage"]:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = []
    for k in range(K):
        v = np.zeros(n)
        run_id = np.cumsum(np.r_[True, bus[k, 1:] != bus[k, :-1]])  # vận tốc chỉ trong 1 xe
        for r in np.unique(run_id[active[k]]):
            m = run_id == r
            if m.sum() > 1:
                v[m] = np.hypot(np.gradient(X[k, m], t_grid[m]), np.gradient(Y[k, m], t_grid[m]))
        xs = pd.Series(X[k]).ffill().bfill().to_numpy()
        ys = pd.Series(Y[k]).ffill().bfill().to_numpy()
        rows.append(pd.DataFrame({
            "window_id": window_id, "vehicle_id": f"slot{k:02d}", "bus_id": bus[k], "line": line[k],
            "t_rel": t_grid, "x_m": xs, "y_m": ys, "v_mps": v, "active": active[k].astype(np.int8),
        }))
    return pd.concat(rows, ignore_index=True)


def build_split(df: pd.DataFrame, center, cfg: dict, date_range) -> pd.DataFrame:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    days = df[(df.ts.dt.normalize() >= start) & (df.ts.dt.normalize() <= end)]
    # lọc thô theo khoảng cách để giảm khối lượng (giữ dư để xe đi vào vùng trong cửa sổ)
    x, y = project_local(days.lat, days.lon, *center)
    days = days[np.hypot(x, y) <= cfg["keep_radius_m"] + 30 * cfg["window_duration_s"]]

    out = []
    for day, df_day in days.groupby(days.ts.dt.normalize()):
        if cfg.get("weekdays_only", True) and day.dayofweek >= 5:
            continue
        for h in cfg["window_start_hours"]:
            wid = f"{day:%Y%m%d}_{h:02d}00"
            w = extract_window(df_day, day + pd.Timedelta(hours=h), center, cfg, wid)
            if not w.empty:
                out.append(w)
    if not out:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(out, ignore_index=True)


# ============================================================ 5. Stats

def residence_times(win: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Thời gian liên tục trong vùng phủ MEC của từng xe buýt; bỏ lượt bị cắt bởi biên
    cửa sổ hoặc biên đoạn dữ liệu (không biết lúc vào/ra thật)."""
    ax, ay = cfg["anchor_xy"]
    res = []
    for _, v in win.groupby(["window_id", "vehicle_id"], sort=False):
        bus = v.bus_id.to_numpy()
        inside = (np.hypot(v.x_m.to_numpy() - ax, v.y_m.to_numpy() - ay) <= cfg["mec_radius_m"]) & (v.active.to_numpy() == 1)
        edges = np.diff(np.r_[0, inside.astype(int), 0])
        for s_, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if s_ > 0 and e < len(inside) and bus[s_ - 1] == bus[s_] and bus[e] == bus[e - 1]:
                res.append((e - s_) * cfg["out_dt"])
    return np.asarray(res)


def neighbor_counts(win: pd.DataFrame, cfg: dict, stride_s: float = 60.0):
    out = {r: [] for r in cfg["neighbor_radii_m"]}
    snap = win[np.isclose(win.t_rel % stride_s, 0.0) & (win.active == 1)]
    for _, s in snap.groupby(["window_id", "t_rel"], sort=False):
        P = s[["x_m", "y_m"]].to_numpy()
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        for r in out:
            out[r].extend(((D <= r).sum(1) - 1).tolist())
    return {r: np.asarray(v) for r, v in out.items()}


def _q(a, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {}
    return {"mean": round(float(a.mean()), 3), **{f"p{int(q * 100)}": round(float(np.quantile(a, q)), 3) for q in qs}}


def compute_stats(clean: pd.DataFrame, splits: dict, cfg: dict, center) -> dict:
    dt = clean.groupby("trip_id").ts.diff().dt.total_seconds().dropna()
    stats = {
        "hotspot_center_latlon": [round(center[0], 6), round(center[1], 6)],
        "sampling_interval_s": _q(dt, (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)),
        "recorded_speed_kmh_all": _q(clean.speed_kmh),
        "stopped_fraction_all": round(float((clean.speed_kmh == 0).mean()), 4),
    }
    for name, win in splits.items():
        if win.empty:
            stats[name] = {"windows": 0}
            continue
        act = win[win.active == 1]
        nb = neighbor_counts(win, cfg)
        stats[name] = {
            "windows": int(win.window_id.nunique()),
            "slots_per_window": int(win.vehicle_id.nunique()),
            "unique_buses": int(act.bus_id.nunique()),
            "lines": int(act.line.nunique()),
            "rows": int(len(win)),
            "slot_coverage": _q(win.groupby(["window_id", "vehicle_id"]).active.mean()),
            "bus_handovers_per_slot_per_window": _q(
                act.groupby(["window_id", "vehicle_id"]).bus_id.nunique() - 1),
            "speed_mps": _q(act.v_mps),
            "stopped_fraction_v_lt_0.5": round(float((act.v_mps < 0.5).mean()), 4),
            "mec_residence_time_s": _q(residence_times(win, cfg)),
            "neighbors": {f"r{int(r)}m": _q(v) for r, v in nb.items()},
        }
    return stats


def plot_overview(clean: pd.DataFrame, splits: dict, cfg: dict, center, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    s = clean.sample(min(len(clean), 300000), random_state=0)
    ax[0].scatter(s.lon, s.lat, s=0.2, alpha=0.2, c="tab:gray")
    ax[0].scatter([center[1]], [center[0]], c="red", s=60, marker="x", label="hotspot / MEC #0")
    ax[0].set(title="Rio bus GPS (sample) & hotspot", xlabel="lon", ylabel="lat"); ax[0].legend()
    win = splits["train"]
    if not win.empty:
        w0 = win[win.window_id == win.window_id.iloc[0]]
        for _, v in w0[w0.active == 1].groupby("bus_id"):
            ax[1].plot(v.x_m, v.y_m, lw=0.8)
        ax[1].add_patch(plt.Circle(cfg["anchor_xy"], cfg["mec_radius_m"], fill=False, color="red", ls="--"))
        ax[1].add_patch(plt.Rectangle((0, 0), *cfg["sim_area"], fill=False, color="k"))
        ax[1].set(title=f"Window {w0.window_id.iloc[0]} (local frame)", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    for name, w in splits.items():
        if not w.empty:
            ax[2].hist(w.v_mps[w.active == 1], bins=60, range=(0, 25), density=True, alpha=0.5, label=name)
    ax[2].set(title="Speed distribution (m/s)", xlabel="v (m/s)"); ax[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ============================================================ main

def main():
    parser = argparse.ArgumentParser(description="Rio Bus GPS Preprocessor")
    parser.add_argument("--config", type=str, default="configs/data/riobus.yaml")
    parser.add_argument("--reuse-clean", action="store_true", help="dùng lại riobus_clean.parquet đã có")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    clean_path, report_path = out_dir / "riobus_clean.parquet", out_dir / "riobus_cleaning_report.json"
    if args.reuse_clean and clean_path.exists() and report_path.exists():
        clean, report = pd.read_parquet(clean_path), json.loads(report_path.read_text())
    else:
        clean, report = clean_raw(load_raw(Path(cfg["raw_dir"])), cfg)
        clean.to_parquet(clean_path, index=False)
        report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Làm sạch: {json.dumps(report)}")

    center = find_hotspot(clean, cfg)
    logger.info(f"Hotspot: lat={center[0]:.6f}, lon={center[1]:.6f}")

    splits = {}
    for name in ("train", "test"):
        splits[name] = build_split(clean, center, cfg, cfg[f"{name}_dates"])
        splits[name].to_parquet(out_dir / f"riobus_{name}.parquet", index=False)
        logger.info(f"{name}: {splits[name].window_id.nunique()} cửa sổ, {len(splits[name])} dòng")

    assert not splits["train"].empty, "Không có cửa sổ train nào đạt min_vehicles_per_window"
    first = splits["train"].window_id.iloc[0]
    splits["train"][splits["train"].window_id == first].to_csv(cfg["compat_csv"], index=False)

    stats = compute_stats(clean, splits, cfg, center)
    stats["cleaning"] = report
    (out_dir / "riobus_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    plot_overview(clean, splits, cfg, center, out_dir / "riobus_overview.png")
    logger.info(f"Xong. Thống kê: {out_dir / 'riobus_stats.json'}")


if __name__ == "__main__":
    main()
