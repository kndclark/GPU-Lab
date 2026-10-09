#!/usr/bin/env bash
# Laptop end of the direct link. NetworkManager, not netplan directly.
# never-default is the load-bearing flag: without it this cable can capture the
# default route and black-hole internet traffic.
# The address is the upper host address of the link's /30 (host/README.md),
# passed in because this repo is public and does not record it.
set -euo pipefail
ADDR=${1:?usage: $0 <laptop link address, the upper host of the /30>}
CON=$(nmcli -t -f NAME,DEVICE con show --active | awk -F: '$2=="enp129s0"{print $1}')
sudo nmcli con mod "$CON" \
    ipv4.method manual \
    ipv4.addresses "$ADDR/30" \
    ipv4.gateway "" \
    ipv4.never-default yes \
    ipv4.route-metric 100 \
    ipv6.method link-local
sudo nmcli con up "$CON"
