# Documentation oracle for the tools a Rocky 9 / Slurm / FreeIPA farm uses
# and this Ubuntu host lacks. research_eval.py runs `<bin> --help` and
# `man <page>` in here (argv only, --network none, read-only) for any eval
# tool not installed on the host, so a lookup answers from the same distro
# the farm will run.
#
#   docker build -t gpu-lab/rocky-docs bench/rocky-docs
FROM rockylinux:9
# The container image installs with tsflags=nodocs; man pages are the point,
# and rpm and dnf came from the base image: upgrading (or else reinstalling)
# them is what brings their pages in. The build fails if they are missing.
RUN sed -i '/^tsflags=nodocs/d' /etc/dnf/dnf.conf \
 && dnf -y install epel-release dnf-plugins-core \
 && dnf config-manager --set-enabled crb \
 && dnf config-manager --add-repo https://rpm.releases.hashicorp.com/RHEL/hashicorp.repo \
 && dnf -y install man-db man-pages \
      slurm ipa-client sssd-tools ansible-core podman nfs-utils xfsprogs \
      firewalld iperf3 traceroute terraform \
 && dnf -y upgrade && (dnf -y reinstall rpm dnf || true) \
 && dnf clean all && mandb -q && man -w rpm dnf sinfo exportfs >/dev/null
ENV MANPAGER=cat PAGER=cat MANWIDTH=100 COLUMNS=100 NO_COLOR=1 TERM=dumb
