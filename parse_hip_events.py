#!/usr/bin/env python3
import sys
import re

def parse(logpath):
    regex = re.compile(r"^.*PROF\s+(\S+)\s+(\S+):\s+([0-9.]+)\s+ms")
    times = {}
    with open(logpath, 'r') as f:
        for l in f:
            m = regex.search(l)
            if m:
                build, name, ms = m.group(1), m.group(2), float(m.group(3))
                times.setdefault(build, {})[name] = times[build].get(name, 0.0) + ms
    return times

def main():
    if len(sys.argv) < 2:
        print('Usage: parse_hip_events.py <logfile>')
        return
    times = parse(sys.argv[1])
    for build, d in times.items():
        total = sum(d.values())
        print(f'BUILD {build} total_ms={total:.3f}')
        for k,v in sorted(d.items(), key=lambda x: -x[1])[:20]:
            print(f'  {k}: {v:.3f} ms  ({v/total*100:.1f}%)')

if __name__ == '__main__':
    main()
