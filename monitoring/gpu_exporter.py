#!/usr/bin/env python3
"""Prometheus exporter for GPU thermals, power and throttle state.

llama-swap's /metrics carries GPU stats only while a model is loaded, so a
training run -- which stops llama-swap to free the card -- is exactly the
window where the existing series go blind. This exporter is independent of
the serving plane for that reason.

Throttle reasons are the point. Temperature alone cannot distinguish a card
running hot but at full clocks from one that is being actively slowed; the
hardware reports that directly and it is otherwise unscraped.
"""
import argparse
import glob
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIELDS = [
    "temperature.gpu",
    "power.draw",
    # power.limit reads N/A on the laptop's GB203M. enforced.power.limit is the
    # one that answers the question: Dynamic Boost floats the ceiling between
    # the 95 W default and the 175 W maximum, so draw alone cannot tell you
    # whether the card is power-limited -- only draw against this can.
    "enforced.power.limit",
    "power.default_limit",
    "power.max_limit",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.graphics",
    "clocks.current.memory",
]

# The driver renames these to clocks_event_reasons.*, but accepts the older
# spelling, which the desktop's driver may still require.
THROTTLE = [
    ("sw_power_cap", "clocks_throttle_reasons.sw_power_cap"),
    ("sw_thermal", "clocks_throttle_reasons.sw_thermal_slowdown"),
    ("hw_thermal", "clocks_throttle_reasons.hw_thermal_slowdown"),
    ("hw_power_brake", "clocks_throttle_reasons.hw_power_brake_slowdown"),
]

GAUGES = [
    ("gpulab_gpu_temperature_celsius", "GPU core temperature"),
    ("gpulab_gpu_power_watts", "Instantaneous board power draw"),
    ("gpulab_gpu_power_limit_watts", "Currently enforced power limit"),
    ("gpulab_gpu_power_limit_default_watts", "Base power limit before Dynamic Boost"),
    ("gpulab_gpu_power_limit_max_watts", "Highest limit the board can be set to"),
    ("gpulab_gpu_memory_used_mib", "Frame buffer memory in use"),
    ("gpulab_gpu_memory_total_mib", "Frame buffer memory total"),
    ("gpulab_gpu_utilization_percent", "Core utilization"),
    ("gpulab_gpu_memory_utilization_percent", "Memory bandwidth utilization"),
    ("gpulab_gpu_clock_graphics_mhz", "Current graphics clock"),
    ("gpulab_gpu_clock_memory_mhz", "Current memory clock"),
]


def sample():
    query = ",".join(FIELDS + [q for _, q in THROTTLE])
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    parts = [p.strip() for p in out.stdout.strip().split("\n")[0].split(",")]
    values, flags = parts[: len(FIELDS)], parts[len(FIELDS) :]

    numbers = []
    for v in values:
        try:
            numbers.append(float(v))
        except ValueError:
            numbers.append(float("nan"))
    throttles = {
        name: 1.0 if flag.strip().lower() == "active" else 0.0
        for (name, _), flag in zip(THROTTLE, flags)
    }
    return numbers, throttles


def render(numbers, throttles, scrape_ok):
    lines = []
    for (name, help_text), value in zip(GAUGES, numbers):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    lines.append("# HELP gpulab_gpu_throttle_active Card is being actively slowed (1=yes)")
    lines.append("# TYPE gpulab_gpu_throttle_active gauge")
    for reason, value in throttles.items():
        lines.append(f'gpulab_gpu_throttle_active{{reason="{reason}"}} {value}')

    lines.append("# HELP gpulab_gpu_scrape_ok Last nvidia-smi poll succeeded")
    lines.append("# TYPE gpulab_gpu_scrape_ok gauge")
    lines.append(f"gpulab_gpu_scrape_ok {scrape_ok}")

    for extra in (system_metrics, link_metrics, foreign_gpu_metrics, textfile_metrics):
        try:
            lines.extend(extra())
        except Exception:
            pass

    return "\n".join(lines) + "\n"


def read_int(path):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def read_str(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except Exception:
        return None


def hwmon_by_name():
    """Map hwmon name -> path. Discovered by name because hwmonN numbering is
    assigned at probe time and moves between boots."""
    found = {}
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        name = read_str(os.path.join(d, "name"))
        if name:
            found.setdefault(name, []).append(d)
    return found


def system_metrics():
    """Fans, CPU package, NVMe and battery.

    On a laptop the CPU and GPU share one vapour chamber, so CPU heat is part of
    the GPU's thermal story rather than a separate concern; fan RPM is the only
    direct read on cooling effort, since nvidia-smi reports fan.speed as N/A on
    this card. Absent sensors are skipped, so the same exporter runs on the
    desktop, which has no battery and no Lenovo WMI.
    """
    lines = []
    mons = hwmon_by_name()

    fans = []
    for d in mons.get("lenovo_wmi_other", []):
        for f in sorted(glob.glob(os.path.join(d, "fan*_input"))):
            rpm = read_int(f)
            if rpm is not None:
                fans.append((os.path.basename(f).replace("_input", ""), rpm))
    if fans:
        lines.append("# HELP gpulab_fan_rpm Chassis fan speed")
        lines.append("# TYPE gpulab_fan_rpm gauge")
        for name, rpm in fans:
            lines.append(f'gpulab_fan_rpm{{fan="{name}"}} {rpm}')

    cores, package = [], None
    for d in mons.get("coretemp", []):
        for f in sorted(glob.glob(os.path.join(d, "temp*_input"))):
            value = read_int(f)
            if value is None:
                continue
            label = read_str(f.replace("_input", "_label")) or ""
            if label.startswith("Package"):
                package = value / 1000.0
            elif label.startswith("Core"):
                cores.append(value / 1000.0)
    if package is not None or cores:
        lines.append("# HELP gpulab_cpu_temperature_celsius CPU temperature")
        lines.append("# TYPE gpulab_cpu_temperature_celsius gauge")
        if package is not None:
            lines.append(f'gpulab_cpu_temperature_celsius{{sensor="package"}} {package}')
        if cores:
            lines.append(f'gpulab_cpu_temperature_celsius{{sensor="core_max"}} {max(cores)}')
            lines.append(f'gpulab_cpu_temperature_celsius{{sensor="core_mean"}} '
                         f'{round(sum(cores)/len(cores), 1)}')

    nvme = []
    for i, d in enumerate(sorted(mons.get("nvme", []))):
        value = read_int(os.path.join(d, "temp1_input"))
        if value is not None:
            nvme.append((f"nvme{i}", value / 1000.0))
    if nvme:
        lines.append("# HELP gpulab_nvme_temperature_celsius NVMe composite temperature")
        lines.append("# TYPE gpulab_nvme_temperature_celsius gauge")
        for name, value in nvme:
            lines.append(f'gpulab_nvme_temperature_celsius{{device="{name}"}} {value}')

    for bat in sorted(glob.glob("/sys/class/power_supply/BAT*")):
        cap = read_int(os.path.join(bat, "capacity"))
        full = read_int(os.path.join(bat, "energy_full")) or read_int(
            os.path.join(bat, "charge_full"))
        design = read_int(os.path.join(bat, "energy_full_design")) or read_int(
            os.path.join(bat, "charge_full_design"))
        cycles = read_int(os.path.join(bat, "cycle_count"))
        power = read_int(os.path.join(bat, "power_now"))
        status = read_str(os.path.join(bat, "status"))
        if cap is not None:
            lines.append("# HELP gpulab_battery_capacity_percent Charge level")
            lines.append("# TYPE gpulab_battery_capacity_percent gauge")
            lines.append(f"gpulab_battery_capacity_percent {cap}")
        if full and design:
            lines.append("# HELP gpulab_battery_health_percent Full charge against design capacity")
            lines.append("# TYPE gpulab_battery_health_percent gauge")
            lines.append(f"gpulab_battery_health_percent {round(100.0*full/design, 2)}")
        if cycles is not None:
            lines.append("# HELP gpulab_battery_cycle_count Charge cycles")
            lines.append("# TYPE gpulab_battery_cycle_count gauge")
            lines.append(f"gpulab_battery_cycle_count {cycles}")
        if power is not None:
            lines.append("# HELP gpulab_battery_power_watts Battery power flow")
            lines.append("# TYPE gpulab_battery_power_watts gauge")
            lines.append(f"gpulab_battery_power_watts {round(power/1e6, 2)}")
        if status is not None:
            lines.append("# HELP gpulab_battery_discharging Battery is discharging (1=yes)")
            lines.append("# TYPE gpulab_battery_discharging gauge")
            lines.append(f"gpulab_battery_discharging {1 if status == 'Discharging' else 0}")
        break

    online = read_int("/sys/class/power_supply/ADP0/online")
    if online is not None:
        lines.append("# HELP gpulab_ac_online AC adapter connected (1=yes)")
        lines.append("# TYPE gpulab_ac_online gauge")
        lines.append(f"gpulab_ac_online {online}")

    return lines


# ---- the alert rules' inputs (monitoring/alerts.yml) ----

# Same names bin/lab uses for the two ends of the direct link.
LINK_NICS = ("enp5s0", "enp129s0")
TEXTFILE_DIR = "/var/lib/gpu-lab/textfile"
# Display servers are unavoidable on the laptop (the panel is on the NVIDIA
# card), so they are the baseline rather than a "foreign" client.
DISPLAY_SERVERS = ("gnome-shell", "Xwayland", "Xorg")


def link_metrics():
    """Direct-link health, as seen from this node.

    carrier_readable=0 with the NIC present is the I226 post-suspend wedge:
    the netdev exists but its carrier file answers EINVAL. up=1 needs carrier
    AND the peer answering a ping, because carrier alone survives a peer whose
    address is gone. A node with neither NIC emits nothing.
    """
    nic = next((n for n in LINK_NICS if os.path.exists(f"/sys/class/net/{n}")), None)
    if nic is None:
        return []
    carrier = read_int(f"/sys/class/net/{nic}/carrier")
    up = 0
    if carrier == 1:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", nic],
                             capture_output=True, text=True, timeout=5).stdout
        peer = {"lab-desktop": "lab-laptop", "lab-laptop": "lab-desktop"}
        target = next((peer[a] for a in peer if f" {a}/" in out), None)
        if target:
            up = 1 if subprocess.run(["ping", "-c1", "-W1", target],
                                     capture_output=True, timeout=5).returncode == 0 else 0
    return [
        "# HELP gpulab_direct_link_up Direct link has carrier and the peer answers ping (1=yes)",
        "# TYPE gpulab_direct_link_up gauge",
        f'gpulab_direct_link_up{{nic="{nic}"}} {up}',
        "# HELP gpulab_direct_link_carrier_readable NIC carrier file is readable (0 = wedged)",
        "# TYPE gpulab_direct_link_carrier_readable gauge",
        f'gpulab_direct_link_carrier_readable{{nic="{nic}"}} {0 if carrier is None else 1}',
    ]


_foreign_cache = (0.0, [])


def foreign_gpu_metrics():
    """Memory held by graphics clients other than the display server.

    vLLM charges any app's growth during its load to its own KV budget (see
    the vllm-kv-charges-other-gpu-clients note), so a browser window on a
    serving card is a KV shrink waiting to happen. Cached 10 s: the XML dump
    is far heavier than the other polls.
    """
    global _foreign_cache
    now = time.time()
    if now - _foreign_cache[0] < 10:
        return _foreign_cache[1]
    import xml.etree.ElementTree as ET
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True,
                         timeout=10, check=True).stdout
    total, count = 0, 0
    for pi in ET.fromstring(xml).iter("process_info"):
        if "G" not in (pi.findtext("type") or ""):
            continue
        name = os.path.basename(pi.findtext("process_name") or "")
        if name in DISPLAY_SERVERS:
            continue
        mem = (pi.findtext("used_memory") or "").split()
        if mem and mem[0].isdigit():
            total += int(mem[0])
            count += 1
    lines = [
        "# HELP gpulab_gpu_foreign_graphics_memory_mib Graphics clients other than the display server",
        "# TYPE gpulab_gpu_foreign_graphics_memory_mib gauge",
        f"gpulab_gpu_foreign_graphics_memory_mib {total}",
        "# HELP gpulab_gpu_foreign_graphics_clients Count of those clients",
        "# TYPE gpulab_gpu_foreign_graphics_clients gauge",
        f"gpulab_gpu_foreign_graphics_clients {count}",
    ]
    _foreign_cache = (now, lines)
    return lines


def textfile_metrics():
    """Serve *.prom files from TEXTFILE_DIR (the canary writes one there)."""
    lines = []
    for f in sorted(glob.glob(os.path.join(TEXTFILE_DIR, "*.prom"))):
        with open(f) as fh:
            lines.extend(ln.rstrip("\n") for ln in fh if ln.strip())
    return lines


class State:
    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.payload = render([float("nan")] * len(GAUGES), {n: 0.0 for n, _ in THROTTLE}, 0)

    def poll_forever(self):
        while True:
            try:
                numbers, throttles = sample()
                payload = render(numbers, throttles, 1)
            except Exception:
                payload = render(
                    [float("nan")] * len(GAUGES), {n: 0.0 for n, _ in THROTTLE}, 0
                )
            with self.lock:
                self.payload = payload
            time.sleep(self.interval)

    def read(self):
        with self.lock:
            return self.payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9835)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    state = State(args.interval)
    threading.Thread(target=state.poll_forever, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/metrics", "/"):
                self.send_error(404)
                return
            body = state.read().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
