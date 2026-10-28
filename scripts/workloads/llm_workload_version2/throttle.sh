#!/usr/bin/env bash
# usage: ./throttle.sh apply | show | clear
F=20971520   # 20 MiB/s
S=5242880    #  5 MiB/s
# 206 and 207 are not in the experiment (207 is in offline_pool; 206 dropped to
# keep F and S equal) and are deliberately left unthrottled.
declare -A RATE=(
  [104]=$F [105]=$F [205]=$F [305]=$F [404]=$F [405]=$F
  [106]=$S [107]=$S [306]=$S [307]=$S [406]=$S [407]=$S
)
SVC=beegfs-storage.service
MODE=${1:-show}

for h in colva1 colva2 colva3 colva4; do
  echo "===== $h"
  case $MODE in
    apply)
      # build "target:rate" list for this host's targets, resolve devices remotely
      args=$(for t in "${!RATE[@]}"; do [[ ${t:0:1} == ${h: -1} ]] && echo "$t:${RATE[$t]}"; done)
      ssh -t "$h" "SVC=$SVC; ARGS='$args'"'
        props=()
        for d in $(sudo grep -oP "^\s*storeStorageDirectory\s*=\s*\K.*" /etc/beegfs/beegfs-storage.conf | tr "," " "); do
          id=$(sudo cat "$d/targetNumID")
          rate=$(printf "%s\n" $ARGS | awk -F: -v id="$id" "\$1==id{print \$2}")
          [[ -z $rate ]] && continue
          src=$(findmnt -no SOURCE -T "$d"); pk=$(lsblk -no PKNAME "$src"); dev=/dev/${pk:-$(basename "$src")}
          echo "  target $id -> $dev  rbps=$rate"
          props+=("IOReadBandwidthMax=$dev $rate")
        done
        # clear first: set-property only adds/overwrites the devices it names,
        # so limits left from an earlier apply (e.g. on 206/207) would survive
        sudo systemctl set-property --runtime "$SVC" IOReadBandwidthMax=
        sudo systemctl set-property --runtime "$SVC" "${props[@]}"
        cat /sys/fs/cgroup/system.slice/$SVC/io.max' ;;
    show)
      ssh "$h" "cat /sys/fs/cgroup/system.slice/$SVC/io.max; lsblk -d -o NAME,MAJ:MIN | grep nvme" ;;
    clear)
      ssh -t "$h" "sudo systemctl set-property --runtime $SVC IOReadBandwidthMax= && cat /sys/fs/cgroup/system.slice/$SVC/io.max" ;;
  esac
done
