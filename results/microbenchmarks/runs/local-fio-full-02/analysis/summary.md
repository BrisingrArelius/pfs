# Local-storage FIO results

Validation: **FAIL**
Parsed measurements: **650**

## Cross-host validation
- ERROR: colva2: scientific configuration differs from colva1 (protocol 4 vs 5)
- ERROR: colva3: scientific configuration differs from colva1 (protocol 4 vs 5)
- ERROR: colva4: scientific configuration differs from colva1 (protocol 4 vs 5)

| Host | Target | Media | Workload | Runs | Median MiB/s | Range MiB/s | Median IOPS | Mean clat ms | p99 clat ms | Stops |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| colva1 | 101 | hdd | rand_read_128k | 5 | 38.0 | 37.5–38.2 | 302 | 409.476 | 1061.159 | 0 byte / 5 time |
| colva1 | 101 | hdd | rand_read_4k | 5 | 1.5 | 1.5–1.5 | 389 | 317.967 | 834.666 | 0 byte / 5 time |
| colva1 | 101 | hdd | rand_write_4k | 5 | 1.5 | 1.5–1.5 | 392 | 315.877 | 943.718 | 0 byte / 5 time |
| colva1 | 101 | hdd | seq_read | 5 | 175.7 | 175.5–176.0 | 174 | 711.054 | 876.610 | 0 byte / 5 time |
| colva1 | 101 | hdd | seq_write | 5 | 184.3 | 183.8–184.4 | 182 | 676.924 | 1182.794 | 0 byte / 5 time |
| colva1 | 103 | hdd | rand_read_128k | 5 | 38.0 | 37.8–38.2 | 302 | 410.097 | 1052.770 | 0 byte / 5 time |
| colva1 | 103 | hdd | rand_read_4k | 5 | 1.5 | 1.5–1.5 | 390 | 317.021 | 843.055 | 0 byte / 5 time |
| colva1 | 103 | hdd | rand_write_4k | 5 | 1.5 | 1.5–1.5 | 389 | 318.275 | 943.718 | 0 byte / 5 time |
| colva1 | 103 | hdd | seq_read | 5 | 180.7 | 180.5–181.0 | 179 | 690.882 | 834.666 | 0 byte / 5 time |
| colva1 | 103 | hdd | seq_write | 5 | 189.0 | 188.9–189.6 | 187 | 659.954 | 1182.794 | 0 byte / 5 time |
| colva1 | 104 | nvme | rand_read_128k | 5 | 4738.2 | 4567.6–5504.8 | 37904 | 3.348 | 10.289 | 0 byte / 5 time |
| colva1 | 104 | nvme | rand_read_4k | 5 | 3975.1 | 3843.9–4001.3 | 1017627 | 0.124 | 0.627 | 0 byte / 5 time |
| colva1 | 104 | nvme | rand_write_4k | 5 | 3970.2 | 3965.2–3971.2 | 1016381 | 0.124 | 0.185 | 0 byte / 5 time |
| colva1 | 104 | nvme | seq_read | 5 | 2226.4 | 2225.1–3949.4 | 2224 | 57.247 | 114.819 | 0 byte / 5 time |
| colva1 | 104 | nvme | seq_write | 5 | 4924.1 | 4913.2–4960.5 | 4922 | 25.838 | 46.399 | 0 byte / 5 time |
| colva1 | 105 | nvme | rand_read_128k | 5 | 4731.3 | 4363.9–5418.9 | 37848 | 3.355 | 10.420 | 0 byte / 5 time |
| colva1 | 105 | nvme | rand_read_4k | 5 | 2971.0 | 2947.1–3769.7 | 760585 | 0.166 | 0.889 | 0 byte / 5 time |
| colva1 | 105 | nvme | rand_write_4k | 5 | 3580.3 | 3567.5–3587.9 | 916559 | 0.138 | 0.200 | 0 byte / 5 time |
| colva1 | 105 | nvme | seq_read | 5 | 2225.5 | 2223.6–3896.7 | 2223 | 57.256 | 115.868 | 0 byte / 5 time |
| colva1 | 105 | nvme | seq_write | 5 | 4706.0 | 4459.2–4754.4 | 4704 | 27.035 | 51.642 | 0 byte / 5 time |
| colva1 | 106 | nvme | rand_read_128k | 5 | 5125.6 | 4253.5–5503.1 | 41003 | 3.101 | 7.242 | 0 byte / 5 time |
| colva1 | 106 | nvme | rand_read_4k | 5 | 3844.0 | 3797.2–3847.8 | 984052 | 0.128 | 0.391 | 0 byte / 5 time |
| colva1 | 106 | nvme | rand_write_4k | 5 | 3590.4 | 3588.2–3604.0 | 919148 | 0.137 | 0.198 | 0 byte / 5 time |
| colva1 | 106 | nvme | seq_read | 5 | 2222.9 | 2219.7–4660.5 | 2221 | 57.346 | 113.770 | 0 byte / 5 time |
| colva1 | 106 | nvme | seq_write | 5 | 4975.4 | 4961.9–4996.8 | 4973 | 25.584 | 46.399 | 0 byte / 5 time |
| colva2 | 201 | hdd | rand_read_128k | 5 | 42.8 | 42.5–43.0 | 342 | 93.380 | 526.385 | 0 byte / 5 time |
| colva2 | 201 | hdd | rand_read_4k | 5 | 1.8 | 1.8–1.8 | 457 | 69.987 | 392.167 | 0 byte / 5 time |
| colva2 | 201 | hdd | rand_write_4k | 5 | 1.7 | 1.7–1.8 | 447 | 71.545 | 92.799 | 0 byte / 5 time |
| colva2 | 201 | hdd | seq_read | 5 | 193.2 | 187.3–195.1 | 193 | 165.402 | 231.735 | 5 byte / 0 time |
| colva2 | 201 | hdd | seq_write | 5 | 196.1 | 195.1–196.6 | 196 | 162.858 | 223.347 | 5 byte / 0 time |
| colva2 | 202 | hdd | rand_read_128k | 5 | 42.7 | 42.5–42.8 | 342 | 93.563 | 541.065 | 0 byte / 5 time |
| colva2 | 202 | hdd | rand_read_4k | 5 | 1.8 | 1.8–1.8 | 455 | 70.226 | 392.167 | 0 byte / 5 time |
| colva2 | 202 | hdd | rand_write_4k | 5 | 1.8 | 1.8–1.8 | 460 | 69.531 | 90.702 | 0 byte / 5 time |
| colva2 | 202 | hdd | seq_read | 5 | 197.6 | 193.4–197.7 | 198 | 161.816 | 225.444 | 5 byte / 0 time |
| colva2 | 202 | hdd | seq_write | 5 | 200.1 | 198.7–200.7 | 200 | 159.591 | 193.987 | 5 byte / 0 time |
| colva2 | 203 | hdd | rand_read_128k | 5 | 43.1 | 43.0–43.2 | 345 | 92.698 | 534.774 | 0 byte / 5 time |
| colva2 | 203 | hdd | rand_read_4k | 5 | 1.8 | 1.8–1.8 | 462 | 69.268 | 375.390 | 0 byte / 5 time |
| colva2 | 203 | hdd | rand_write_4k | 5 | 1.7 | 1.7–1.7 | 439 | 72.813 | 94.896 | 0 byte / 5 time |
| colva2 | 203 | hdd | seq_read | 5 | 189.4 | 185.7–191.0 | 189 | 168.766 | 212.861 | 5 byte / 0 time |
| colva2 | 203 | hdd | seq_write | 5 | 192.9 | 192.3–193.6 | 193 | 165.572 | 229.638 | 5 byte / 0 time |
| colva2 | 204 | hdd | rand_read_128k | 5 | 41.1 | 41.0–41.3 | 329 | 97.286 | 541.065 | 0 byte / 5 time |
| colva2 | 204 | hdd | rand_read_4k | 5 | 1.7 | 1.7–1.8 | 448 | 71.432 | 387.973 | 0 byte / 5 time |
| colva2 | 204 | hdd | rand_write_4k | 5 | 1.7 | 1.7–1.7 | 431 | 74.240 | 109.576 | 0 byte / 5 time |
| colva2 | 204 | hdd | seq_read | 5 | 167.0 | 156.0–167.5 | 167 | 191.429 | 254.804 | 0 byte / 5 time |
| colva2 | 204 | hdd | seq_write | 5 | 168.8 | 166.8–168.9 | 169 | 189.217 | 238.027 | 0 byte / 5 time |
| colva2 | 205 | nvme | rand_read_128k | 5 | 5400.8 | 5389.5–5415.1 | 43207 | 0.732 | 3.293 | 5 byte / 0 time |
| colva2 | 205 | nvme | rand_read_4k | 5 | 2130.2 | 2120.1–2137.3 | 545338 | 0.057 | 0.105 | 5 byte / 0 time |
| colva2 | 205 | nvme | rand_write_4k | 5 | 1356.1 | 1354.1–1365.2 | 347165 | 0.090 | 0.118 | 5 byte / 0 time |
| colva2 | 205 | nvme | seq_read | 5 | 2213.1 | 2212.6–6251.5 | 2213 | 14.340 | 14.615 | 5 byte / 0 time |
| colva2 | 205 | nvme | seq_write | 5 | 4675.8 | 4671.5–4701.6 | 4676 | 6.754 | 10.945 | 5 byte / 0 time |
| colva2 | 206 | nvme | rand_read_128k | 5 | 5398.0 | 5378.2–5406.5 | 43184 | 0.732 | 3.293 | 5 byte / 0 time |
| colva2 | 206 | nvme | rand_read_4k | 5 | 2143.2 | 2137.8–2144.5 | 548648 | 0.057 | 0.106 | 5 byte / 0 time |
| colva2 | 206 | nvme | rand_write_4k | 5 | 1365.9 | 1361.7–1375.2 | 349665 | 0.089 | 0.117 | 5 byte / 0 time |
| colva2 | 206 | nvme | seq_read | 5 | 2212.6 | 2212.6–6198.5 | 2213 | 14.343 | 14.615 | 5 byte / 0 time |
| colva2 | 206 | nvme | seq_write | 5 | 4671.5 | 4665.1–4682.2 | 4672 | 6.768 | 11.207 | 5 byte / 0 time |
| colva2 | 207 | nvme | rand_read_128k | 5 | 5392.3 | 5383.8–5429.5 | 43138 | 0.732 | 3.293 | 5 byte / 0 time |
| colva2 | 207 | nvme | rand_read_4k | 5 | 2121.4 | 2112.2–2124.0 | 543079 | 0.057 | 0.105 | 5 byte / 0 time |
| colva2 | 207 | nvme | rand_write_4k | 5 | 1352.7 | 1338.9–1356.7 | 346293 | 0.090 | 0.118 | 5 byte / 0 time |
| colva2 | 207 | nvme | seq_read | 5 | 2213.1 | 2209.8–6228.7 | 2213 | 14.354 | 14.615 | 5 byte / 0 time |
| colva2 | 207 | nvme | seq_write | 5 | 4677.9 | 4660.9–4682.2 | 4678 | 6.757 | 11.207 | 5 byte / 0 time |
| colva3 | 301 | hdd | rand_read_128k | 5 | 42.8 | 42.6–43.0 | 342 | 93.534 | 517.997 | 0 byte / 5 time |
| colva3 | 301 | hdd | rand_read_4k | 5 | 1.8 | 1.8–1.8 | 455 | 70.341 | 387.973 | 0 byte / 5 time |
| colva3 | 301 | hdd | rand_write_4k | 5 | 1.7 | 1.7–1.7 | 429 | 74.517 | 104.333 | 0 byte / 5 time |
| colva3 | 301 | hdd | seq_read | 5 | 190.6 | 189.4–192.3 | 191 | 167.833 | 225.444 | 5 byte / 0 time |
| colva3 | 301 | hdd | seq_write | 5 | 193.3 | 192.3–193.9 | 193 | 165.287 | 231.735 | 5 byte / 0 time |
| colva3 | 302 | hdd | rand_read_128k | 5 | 42.8 | 42.4–42.9 | 343 | 93.379 | 526.385 | 0 byte / 5 time |
| colva3 | 302 | hdd | rand_read_4k | 5 | 1.8 | 1.8–1.8 | 456 | 70.225 | 387.973 | 0 byte / 5 time |
| colva3 | 302 | hdd | rand_write_4k | 5 | 1.7 | 1.7–1.7 | 431 | 74.263 | 98.042 | 0 byte / 5 time |
| colva3 | 302 | hdd | seq_read | 5 | 193.8 | 185.5–194.5 | 194 | 165.000 | 225.444 | 5 byte / 0 time |
| colva3 | 302 | hdd | seq_write | 5 | 196.1 | 194.5–196.9 | 196 | 162.998 | 219.152 | 5 byte / 0 time |
| colva3 | 303 | hdd | rand_read_128k | 5 | 42.4 | 42.3–42.5 | 339 | 94.267 | 534.774 | 0 byte / 5 time |
| colva3 | 303 | hdd | rand_read_4k | 5 | 1.8 | 1.8–1.8 | 452 | 70.767 | 387.973 | 0 byte / 5 time |
| colva3 | 303 | hdd | rand_write_4k | 5 | 1.7 | 1.7–1.7 | 433 | 73.969 | 98.042 | 0 byte / 5 time |
| colva3 | 303 | hdd | seq_read | 5 | 188.2 | 185.2–188.8 | 188 | 169.924 | 283.116 | 5 byte / 0 time |
| colva3 | 303 | hdd | seq_write | 5 | 191.2 | 190.6–191.8 | 191 | 167.101 | 221.250 | 5 byte / 0 time |
| colva3 | 304 | hdd | rand_read_128k | 5 | 40.2 | 39.8–40.8 | 321 | 99.570 | 566.231 | 0 byte / 5 time |
| colva3 | 304 | hdd | rand_read_4k | 5 | 1.7 | 1.7–1.7 | 434 | 73.714 | 408.945 | 0 byte / 5 time |
| colva3 | 304 | hdd | rand_write_4k | 5 | 1.7 | 1.7–1.7 | 429 | 74.574 | 101.188 | 0 byte / 5 time |
| colva3 | 304 | hdd | seq_read | 5 | 161.6 | 157.1–167.7 | 162 | 197.873 | 383.779 | 0 byte / 5 time |
| colva3 | 304 | hdd | seq_write | 5 | 168.8 | 168.3–169.2 | 169 | 189.356 | 252.707 | 0 byte / 5 time |
| colva3 | 305 | nvme | rand_read_128k | 5 | 5426.6 | 5400.8–5464.2 | 43413 | 0.734 | 3.391 | 5 byte / 0 time |
| colva3 | 305 | nvme | rand_read_4k | 5 | 2163.1 | 2140.5–2170.4 | 553747 | 0.056 | 0.104 | 5 byte / 0 time |
| colva3 | 305 | nvme | rand_write_4k | 5 | 1375.6 | 1370.4–1378.4 | 352155 | 0.088 | 0.113 | 5 byte / 0 time |
| colva3 | 305 | nvme | seq_read | 5 | 2214.1 | 2211.7–6251.5 | 2214 | 14.428 | 14.615 | 5 byte / 0 time |
| colva3 | 305 | nvme | seq_write | 5 | 4650.3 | 4639.8–4663.0 | 4650 | 6.840 | 11.076 | 5 byte / 0 time |
| colva3 | 306 | nvme | rand_read_128k | 5 | 5412.3 | 5400.8–5432.4 | 43298 | 0.735 | 3.326 | 5 byte / 0 time |
| colva3 | 306 | nvme | rand_read_4k | 5 | 2159.9 | 2158.5–2162.6 | 552930 | 0.056 | 0.105 | 5 byte / 0 time |
| colva3 | 306 | nvme | rand_write_4k | 5 | 1384.3 | 1375.6–1388.7 | 354392 | 0.088 | 0.112 | 5 byte / 0 time |
| colva3 | 306 | nvme | seq_read | 5 | 2213.6 | 2212.1–6232.5 | 2214 | 14.431 | 14.615 | 5 byte / 0 time |
| colva3 | 306 | nvme | seq_write | 5 | 4656.7 | 4633.5–4669.4 | 4657 | 6.830 | 11.076 | 5 byte / 0 time |
| colva3 | 307 | nvme | rand_read_128k | 5 | 5406.5 | 5398.0–5423.7 | 43252 | 0.736 | 3.359 | 5 byte / 0 time |
| colva3 | 307 | nvme | rand_read_4k | 5 | 2139.1 | 2129.3–2146.8 | 547616 | 0.057 | 0.104 | 5 byte / 0 time |
| colva3 | 307 | nvme | rand_write_4k | 5 | 1369.7 | 1365.2–1374.5 | 350647 | 0.089 | 0.113 | 5 byte / 0 time |
| colva3 | 307 | nvme | seq_read | 5 | 2213.1 | 2212.1–6270.7 | 2213 | 14.432 | 14.615 | 5 byte / 0 time |
| colva3 | 307 | nvme | seq_write | 5 | 4658.8 | 4614.7–4671.5 | 4659 | 6.829 | 11.469 | 5 byte / 0 time |
| colva4 | 401 | hdd | rand_read_128k | 5 | 41.2 | 41.0–41.5 | 330 | 96.932 | 541.065 | 0 byte / 5 time |
| colva4 | 401 | hdd | rand_read_4k | 5 | 1.8 | 1.7–1.8 | 453 | 70.618 | 383.779 | 0 byte / 5 time |
| colva4 | 401 | hdd | rand_write_4k | 5 | 1.8 | 1.8–1.8 | 462 | 69.238 | 90.702 | 0 byte / 5 time |
| colva4 | 401 | hdd | seq_read | 5 | 165.6 | 158.4–166.1 | 166 | 193.009 | 231.735 | 0 byte / 5 time |
| colva4 | 401 | hdd | seq_write | 5 | 166.7 | 165.8–167.3 | 167 | 191.811 | 252.707 | 0 byte / 5 time |
| colva4 | 402 | hdd | rand_read_128k | 5 | 42.3 | 42.2–42.5 | 338 | 94.433 | 530.579 | 0 byte / 5 time |
| colva4 | 402 | hdd | rand_read_4k | 5 | 1.8 | 1.8–1.8 | 455 | 70.291 | 387.973 | 0 byte / 5 time |
| colva4 | 402 | hdd | rand_write_4k | 5 | 1.9 | 1.9–1.9 | 478 | 66.815 | 88.605 | 0 byte / 5 time |
| colva4 | 402 | hdd | seq_read | 5 | 182.2 | 172.2–183.1 | 182 | 175.486 | 223.347 | 5 byte / 0 time |
| colva4 | 402 | hdd | seq_write | 5 | 184.4 | 183.7–185.1 | 184 | 173.362 | 227.541 | 5 byte / 0 time |
| colva4 | 403 | hdd | rand_read_128k | 5 | 38.8 | 38.7–39.1 | 310 | 103.042 | 574.620 | 0 byte / 5 time |
| colva4 | 403 | hdd | rand_read_4k | 5 | 1.7 | 1.7–1.7 | 440 | 72.622 | 400.556 | 0 byte / 5 time |
| colva4 | 403 | hdd | rand_write_4k | 5 | 1.8 | 1.8–1.8 | 455 | 70.198 | 98.042 | 0 byte / 5 time |
| colva4 | 403 | hdd | seq_read | 5 | 143.7 | 142.9–144.5 | 144 | 222.450 | 287.310 | 0 byte / 5 time |
| colva4 | 403 | hdd | seq_write | 5 | 144.8 | 143.7–145.1 | 145 | 220.744 | 312.476 | 0 byte / 5 time |
| colva4 | 404 | nvme | rand_read_128k | 5 | 5369.7 | 5341.7–5398.0 | 42958 | 0.736 | 3.391 | 5 byte / 0 time |
| colva4 | 404 | nvme | rand_read_4k | 5 | 1613.6 | 1607.8–1615.1 | 413085 | 0.075 | 0.113 | 5 byte / 0 time |
| colva4 | 404 | nvme | rand_write_4k | 5 | 1179.2 | 1176.6–1182.6 | 301870 | 0.103 | 0.127 | 5 byte / 0 time |
| colva4 | 404 | nvme | seq_read | 5 | 2212.6 | 2210.2–6534.8 | 2213 | 14.271 | 14.483 | 5 byte / 0 time |
| colva4 | 404 | nvme | seq_write | 5 | 4677.9 | 4663.0–4880.8 | 4678 | 6.753 | 10.945 | 5 byte / 0 time |
| colva4 | 405 | nvme | rand_read_128k | 5 | 5361.3 | 5347.3–5467.2 | 42890 | 0.736 | 3.293 | 5 byte / 0 time |
| colva4 | 405 | nvme | rand_read_4k | 5 | 1632.7 | 1625.7–1635.0 | 417959 | 0.075 | 0.112 | 5 byte / 0 time |
| colva4 | 405 | nvme | rand_write_4k | 5 | 1173.4 | 1170.6–1179.9 | 300383 | 0.104 | 0.132 | 5 byte / 0 time |
| colva4 | 405 | nvme | seq_read | 5 | 2213.1 | 2212.6–6653.7 | 2213 | 14.270 | 14.483 | 5 byte / 0 time |
| colva4 | 405 | nvme | seq_write | 5 | 4663.0 | 4637.7–4873.9 | 4663 | 6.776 | 11.338 | 5 byte / 0 time |
| colva4 | 406 | nvme | rand_read_128k | 5 | 5364.1 | 5350.1–5375.3 | 42913 | 0.735 | 3.359 | 5 byte / 0 time |
| colva4 | 406 | nvme | rand_read_4k | 5 | 1631.1 | 1627.7–1633.2 | 417560 | 0.075 | 0.111 | 5 byte / 0 time |
| colva4 | 406 | nvme | rand_write_4k | 5 | 1174.0 | 1169.2–1176.2 | 300555 | 0.104 | 0.134 | 5 byte / 0 time |
| colva4 | 406 | nvme | seq_read | 5 | 2213.1 | 2212.6–6653.7 | 2213 | 14.264 | 14.483 | 5 byte / 0 time |
| colva4 | 406 | nvme | seq_write | 5 | 4671.5 | 4635.6–4939.7 | 4672 | 6.763 | 11.076 | 5 byte / 0 time |
| colva4 | 407 | nvme | rand_read_128k | 5 | 5372.5 | 5338.9–5443.9 | 42980 | 0.735 | 3.424 | 5 byte / 0 time |
| colva4 | 407 | nvme | rand_read_4k | 5 | 1622.6 | 1618.2–1629.0 | 415376 | 0.075 | 0.111 | 5 byte / 0 time |
| colva4 | 407 | nvme | rand_write_4k | 5 | 1168.9 | 1165.0–1172.6 | 299251 | 0.104 | 0.134 | 5 byte / 0 time |
| colva4 | 407 | nvme | seq_read | 5 | 2213.1 | 2212.1–6653.7 | 2213 | 14.268 | 14.483 | 5 byte / 0 time |
| colva4 | 407 | nvme | seq_write | 5 | 4680.1 | 4650.3–4946.9 | 4680 | 6.748 | 11.207 | 5 byte / 0 time |

## colva1 validation
- ERROR: target 102: cleanup is pending
- ERROR: target 102: no completed preparation
- ERROR: target 107: cleanup is pending
- ERROR: target 107: no completed preparation
- ERROR: 102__rand_read_128k__r1: latest attempt is not completed
- ERROR: 102__rand_read_4k__r1: latest attempt is not completed
- ERROR: 102__rand_write_4k__r1: latest attempt is not completed
- ERROR: 102__seq_read__r1: latest attempt is not completed
- ERROR: 102__seq_write__r1: latest attempt is not completed
- ERROR: 102__rand_read_4k__r2: latest attempt is not completed
- ERROR: 102__rand_write_4k__r2: latest attempt is not completed
- ERROR: 102__seq_read__r2: latest attempt is not completed
- ERROR: 102__seq_write__r2: latest attempt is not completed
- ERROR: 102__rand_read_128k__r2: latest attempt is not completed
- ERROR: 102__rand_write_4k__r3: latest attempt is not completed
- ERROR: 102__seq_read__r3: latest attempt is not completed
- ERROR: 102__seq_write__r3: latest attempt is not completed
- ERROR: 102__rand_read_128k__r3: latest attempt is not completed
- ERROR: 102__rand_read_4k__r3: latest attempt is not completed
- ERROR: 102__seq_read__r4: latest attempt is not completed
- ERROR: 102__seq_write__r4: latest attempt is not completed
- ERROR: 102__rand_read_128k__r4: latest attempt is not completed
- ERROR: 102__rand_read_4k__r4: latest attempt is not completed
- ERROR: 102__rand_write_4k__r4: latest attempt is not completed
- ERROR: 102__seq_write__r5: latest attempt is not completed
- ERROR: 102__rand_read_128k__r5: latest attempt is not completed
- ERROR: 102__rand_read_4k__r5: latest attempt is not completed
- ERROR: 102__rand_write_4k__r5: latest attempt is not completed
- ERROR: 102__seq_read__r5: latest attempt is not completed
- ERROR: 107__rand_read_128k__r1: latest attempt is not completed
- ERROR: 107__rand_read_4k__r1: latest attempt is not completed
- ERROR: 107__rand_write_4k__r1: latest attempt is not completed
- ERROR: 107__seq_read__r1: latest attempt is not completed
- ERROR: 107__seq_write__r1: latest attempt is not completed
- ERROR: 107__rand_read_4k__r2: latest attempt is not completed
- ERROR: 107__rand_write_4k__r2: latest attempt is not completed
- ERROR: 107__seq_read__r2: latest attempt is not completed
- ERROR: 107__seq_write__r2: latest attempt is not completed
- ERROR: 107__rand_read_128k__r2: latest attempt is not completed
- ERROR: 107__rand_write_4k__r3: latest attempt is not completed
- ERROR: 107__seq_read__r3: latest attempt is not completed
- ERROR: 107__seq_write__r3: latest attempt is not completed
- ERROR: 107__rand_read_128k__r3: latest attempt is not completed
- ERROR: 107__rand_read_4k__r3: latest attempt is not completed
- ERROR: 107__seq_read__r4: latest attempt is not completed
- ERROR: 107__seq_write__r4: latest attempt is not completed
- ERROR: 107__rand_read_128k__r4: latest attempt is not completed
- ERROR: 107__rand_read_4k__r4: latest attempt is not completed
- ERROR: 107__rand_write_4k__r4: latest attempt is not completed
- ERROR: 107__seq_write__r5: latest attempt is not completed
- ERROR: 107__rand_read_128k__r5: latest attempt is not completed
- ERROR: 107__rand_read_4k__r5: latest attempt is not completed
- ERROR: 107__rand_write_4k__r5: latest attempt is not completed
- ERROR: 107__seq_read__r5: latest attempt is not completed
- WARNING: latest session outcome is failed
