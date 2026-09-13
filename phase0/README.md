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
