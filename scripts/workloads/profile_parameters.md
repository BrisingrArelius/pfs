read ratio(RR) = POSIX_READS / (POSIX_READS + POSIX_WRITES)
RR>0.5 => read_heavy
RR<0.5 => write_heavy
RR=0.5 => mixed