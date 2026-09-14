#!/usr/bin/env bash
# Laptop only. Repair the direct-link NIC after resume from suspend.
#
# The Intel I226-V (igc, PCI 0000:81:00.0) does not reliably survive a deep
# suspend. The failure is not a dead cable and not a NetworkManager problem:
# the kernel detaches the device on the way down and the resume path fails
# (`Timeout reading IGC_PTM_STAT register`). What is left is a netdev that
# still appears in `ip link show` but rejects every operation with ENODEV:
#
#     $ sudo ip link set enp129s0 up
#     RTNETLINK answers: No such device
#
# With the laptop's PHY powered down the desktop sees no carrier either, so
# systemd-networkd never applies 10.10.0.1 and the whole lab looks unreachable
# from an entirely healthy server. Rebinding the PCI driver re-probes the
# device; NetworkManager then reapplies 10.10.0.2/30 without help.
#
# Run by gpu-lab-igc-resume.service on resume. Safe to run by hand, and a no-op
# on any machine without this NIC.
set -uo pipefail

IFACE=${1:-enp129s0}
DRIVER=igc
FALLBACK_PCI=0000:81:00.0
SETTLE_TRIES=10          # seconds to let the NIC come back on its own
REAPPEAR_TRIES=15        # seconds to wait for the netdev after rebinding
LINK_TRIES=15            # seconds to wait for 2.5GbE to negotiate
ADDR_TRIES=10            # seconds to wait for NetworkManager to reapply the address

log() { printf '%s\n' "$*"; }        # stdout and stderr land in the journal

# Healthy means ethtool can talk to the device. That is still true with the
# cable unplugged ("Link detected: no"), so this tests the device, not the link.
healthy() { ethtool "$IFACE" >/dev/null 2>&1; }
carrier() { [ "$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null)" = 1 ]; }

pci_addr() {
    local d
    d=$(readlink -f "/sys/class/net/$IFACE/device" 2>/dev/null) && [ -n "$d" ] \
        && { basename "$d"; return 0; }
    # netdev gone entirely: fall back to whatever this driver still holds
    for d in /sys/bus/pci/drivers/$DRIVER/0000:*; do
        [ -e "$d" ] && { basename "$d"; return 0; }
    done
    [ -e "/sys/bus/pci/devices/$FALLBACK_PCI" ] && { echo "$FALLBACK_PCI"; return 0; }
    return 1
}

# Not this machine, or no such NIC: do nothing at all.
if [ ! -e "/sys/class/net/$IFACE" ] && [ ! -e "/sys/bus/pci/devices/$FALLBACK_PCI" ]; then
    log "no $IFACE and no $FALLBACK_PCI on this host - nothing to do"
    exit 0
fi

for _ in $(seq 1 "$SETTLE_TRIES"); do
    healthy && { log "$IFACE came back on its own - no repair needed"; exit 0; }
    sleep 1
done

PCI=$(pci_addr) || { log "ERROR: $IFACE is wedged and no PCI address could be resolved"; exit 1; }
log "$IFACE did not survive suspend (ENODEV after ${SETTLE_TRIES}s) - rebinding $DRIVER at $PCI"

if [ -w "/sys/bus/pci/drivers/$DRIVER/unbind" ]; then
    echo "$PCI" > "/sys/bus/pci/drivers/$DRIVER/unbind" 2>/dev/null \
        || log "note: unbind reported an error (already unbound?), continuing"
    sleep 2
fi
echo "$PCI" > "/sys/bus/pci/drivers/$DRIVER/bind" 2>/dev/null \
    || log "note: bind reported an error, continuing"

for _ in $(seq 1 "$REAPPEAR_TRIES"); do healthy && break; sleep 1; done

if ! healthy; then
    log "ERROR: $IFACE still unusable after rebinding $PCI - a reboot is the fallback"
    exit 1
fi

# The device is back, but 2.5GbE negotiation takes a few seconds and
# NetworkManager only reapplies 10.10.0.2/30 once carrier appears. Wait for the
# real end state before logging: the journal is the only evidence anyone gets
# after an unattended resume, and a premature line there reads like a failure.
for _ in $(seq 1 "$LINK_TRIES"); do carrier && break; sleep 1; done

if ! carrier; then
    # Not an error. The NIC is fixed; there is simply nothing on the other end.
    log "$IFACE repaired, but no carrier after ${LINK_TRIES}s - cable unplugged, or the desktop is down"
    exit 0
fi

for _ in $(seq 1 "$ADDR_TRIES"); do
    ip -4 addr show "$IFACE" 2>/dev/null | grep -q 'inet ' && break
    sleep 1
done

log "repaired: $(ethtool "$IFACE" 2>/dev/null | awk -F': ' '/Speed:|Duplex:/ {printf "%s ", $2}')"
log "address:  $(ip -br addr show "$IFACE" 2>/dev/null | tr -s ' ')"
exit 0
