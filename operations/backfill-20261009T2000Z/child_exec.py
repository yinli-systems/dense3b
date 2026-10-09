"""One isolated assertion check; called only inside fresh user/network/PID namespaces."""
import os,sys,resource
from pathlib import Path
from isolation import restrict
program=Path(sys.argv[1]).read_text()
resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30))
resource.setrlimit(resource.RLIMIT_CPU,(10,10))
resource.setrlimit(resource.RLIMIT_FSIZE,(8<<20,8<<20))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
resource.setrlimit(resource.RLIMIT_NPROC,(64,64))
restrict()
exec(compile(program,'candidate.py','exec'),{'__name__':'__main__'})
