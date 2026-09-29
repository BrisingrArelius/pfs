import darshan
darshan_file = '/home/pfs/advay/pfs/13721-10081698891450254892-zlib.darshan'
# Load a log file (automatically decompresses .darshan or .gz)
report = darshan.DarshanReport(darshan_file, read_all=False)

# Print high-level job summary
print(f"Job ID: {report.metadata['job']['jobid']}")
print(f"Number of Ranks: {report.metadata['job']['nprocs']}")
print(f"Run Time (s): {report.metadata['job']['run_time']}")
print(f"Loaded Modules: {list(report.records.keys())}")