# Phase 0 — host configuration

Everything here lives *outside* the repo on the two machines, and is not
reproducible from the code alone. It is recorded so a reinstall does not mean
rediscovering it.

**Nothing in this directory contains a secret.** The Wi-Fi stanzas of both
machines' network config are deliberately omitted — the desktop's
`/etc/netplan/00-installer-config.yaml` carries a PSK in cleartext. Only the
direct-link ethernet configuration is captured.

## The direct link

Both nodes had an idle 2.5GbE port and were talking over Wi-Fi. Measured between
the same two machines:

| Path | RTT |
|---|---|
| Wi-Fi | 34–187 ms, avg 103 — the *spread* is the problem, not the average |
| Direct cable | 0.55 ms |

A private /30 with no gateway. Two rules matter:

- **No default route on this link.** `ipv4.never-default yes` on the laptop's
  NetworkManager profile; no `gateway4` in the desktop's netplan. Without this the
  cable can win the routing contest and send internet traffic into a dead end.
- **The desktop's netplan matches on MAC**, so the interface name cannot drift.

Addresses: desktop `10.10.0.1/30` (`enp5s0`), laptop `10.10.0.2/30` (`enp129s0`).

## The laptop's NIC does not survive suspend

The laptop's Intel I226-V (`igc`, PCI `0000:81:00.0`) is detached by the kernel
on the way into deep suspend and its resume path fails
(`Timeout reading IGC_PTM_STAT register`). What is left is a netdev that still
appears in `ip link show` but rejects every operation:

    $ sudo ip link set enp129s0 up
    RTNETLINK answers: No such device

**This reads as a desktop failure and is not one.** With the laptop's PHY
powered down the desktop sees no carrier either, so systemd-networkd never
applies `10.10.0.1` — `ip -br addr` there shows `enp5s0 DOWN` with no address at
all. That missing address is a *symptom*, not lost configuration; networkd does
not address a link with no carrier. Check `/etc/netplan/` before re-applying
anything, and reach the desktop over `ssh llm-wifi` meanwhile.

Rebinding the PCI driver re-probes the device; NetworkManager then reapplies
`10.10.0.2/30` on its own. That is automated by `gpu-lab-igc-resume.service`,
installed from the two files here:

| Repo file | Installed as |
|---|---|
| `laptop-igc-resume-repair.sh` | `/usr/local/sbin/gpu-lab-igc-resume-repair` (0755 root) |
| `laptop-igc-resume.service` | `/etc/systemd/system/gpu-lab-igc-resume.service`, `systemctl enable` |

Three choices in it are deliberate:

- **A unit ordered `Before=sleep.target` with the work in `ExecStop=`**, not a
  script in `/usr/lib/systemd/system-sleep/`. Sleep-directory scripts run with
  `user.slice` frozen and *block* the resume until they return, and this one
  waits up to ten seconds to see whether the NIC recovers unaided.
- **`WantedBy=sleep.target`** rather than naming `suspend.target` — `sleep.target`
  is pulled in by all four sleep types, so one unit covers suspend, hibernate,
  hybrid-sleep and suspend-then-hibernate.
- **It repairs only when the NIC is actually wedged.** The probe is whether
  `ethtool` can talk to the device, which stays true with the cable unplugged,
  so it tests the device rather than the link. A healthy resume logs one line
  and exits.

No carrier after a successful rebind is reported, not treated as an error — that
is the legitimate "desktop is off" case. Run it by hand any time:
`sudo /usr/local/sbin/gpu-lab-igc-resume-repair`. `bin/lab status` reports the
link on either node.

## The shared model cache

`/srv/model-cache`, exported from the desktop over NFSv4.2. This is a
*correctness* requirement, not a convenience: a cross-architecture comparison is
only meaningful if both nodes loaded byte-identical weights.

Two non-obvious choices in the export line:

- **`10.10.0.2/32`** — exported to the laptop's direct-link address only. NFS
  listens on all interfaces, so this ACL is what keeps it off Wi-Fi.
- **`all_squash,anonuid=1000,anongid=1000`** — every write lands as uid 1000
  regardless of what UID runs inside a container. Without it, a container running
  as root writes files neither machine can manage. Both nodes have david at
  uid/gid 1000; check that before reusing this.

Measured over the direct link: **294 MB/s read** (94% of 2.5GbE line rate),
189 MB/s write. A 20 GB model crosses in ~68 s.

The write side is capped by the `sync` export option. `async` would raise it at
the usual crash-consistency cost — defensible here only because these weights are
re-downloadable.

On the laptop, `nofail,x-systemd.automount` means an unplugged cable degrades to a
missing directory rather than a failed boot.

## Driver notes

Both nodes are on **595.91.07**, reached by convergence rather than pinning. That
is not the same as a pin: the next `apt upgrade` on one node and not the other
reopens the gap. Before a benchmark sweep, `apt-mark hold` the driver
metapackages on both.

The laptop runs the **DKMS** driver (`nvidia-driver-595-open`) and has a MOK
enrolled under Secure Boot, because its display is wired to the NVIDIA GPU and it
needs the full GL/EGL/GBM stack. The desktop runs the **precompiled headless**
variant (`nvidia-headless-no-dkms-595-server-open`) and needs no MOK — Canonical's
modules are already signed by a key the shim trusts.

Do not run `ubuntu-drivers install --gpgpu` on the laptop. It selects the headless
line, strips `libnvidia-gl` / `xserver-xorg-video-nvidia`, and the next boot has no
driver for the GPU the panel is attached to. Confirm which GPU owns the display
before trusting any driver flag:

    for c in /sys/class/drm/card*-*/; do
      echo "$(basename $c) $(cat $c/status) $(basename $(readlink -f $c/../device/driver))"
    done
