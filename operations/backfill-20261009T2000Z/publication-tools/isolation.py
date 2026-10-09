"""Unprivileged, fail-closed filesystem and syscall restrictions for Python test children.
Landlock ABI 1 plus a seccomp allowlist; run the coordinator in new user/net/PID namespaces.
No privileged mounts, global configuration, or training filesystem changes are required.
"""
import ctypes,ctypes.util,errno,os,platform,resource,sysconfig
from pathlib import Path
LIBC=ctypes.CDLL(None,use_errno=True)
class Rules(ctypes.Structure):_fields_=[('handled',ctypes.c_uint64)]
class Beneath(ctypes.Structure):
 _pack_=1
 _fields_=[('allowed',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
# Python/numpy execution and existing IPC descriptors; no new sockets or privilege/admin syscalls.
ALLOWED='read write open openat close stat fstat lstat newfstatat statx access faccessat faccessat2 readlink readlinkat lseek pread64 pwrite64 readv writev preadv pwritev preadv2 pwritev2 mmap mprotect munmap mremap madvise brk msync mincore getrandom getpid getppid gettid getuid geteuid getgid getegid getgroups set_tid_address set_robust_list rseq arch_prctl futex futex_waitv rt_sigaction rt_sigprocmask rt_sigreturn rt_sigpending rt_sigtimedwait rt_sigsuspend sigaltstack rt_sigqueueinfo rt_tgsigqueueinfo getitimer setitimer alarm timer_create timer_settime timer_gettime timer_delete clone clone3 fork vfork wait4 waitid exit exit_group kill tkill tgkill sched_yield sched_getaffinity sched_setaffinity sched_getparam sched_getscheduler sched_get_priority_max sched_get_priority_min nanosleep clock_nanosleep clock_gettime clock_getres gettimeofday time times getrusage uname sysinfo getrlimit setrlimit prlimit64 getcwd chdir fchdir getdents getdents64 umask fcntl flock ioctl dup dup2 dup3 pipe pipe2 eventfd eventfd2 poll ppoll select pselect6 epoll_create epoll_create1 epoll_ctl epoll_wait epoll_pwait epoll_pwait2 fsync fdatasync ftruncate mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2 link linkat symlink symlinkat memfd_create restart_syscall'.split()
def _check(ret,label):
 if ret<0:raise OSError(ctypes.get_errno(),label)
 return ret
def restrict():
 if platform.machine()!='x86_64':raise RuntimeError('Reviewed syscall mapping is x86_64 only')
 abi=_check(LIBC.syscall(444,0,0,1),'Landlock unavailable')
 if abi<1:raise RuntimeError('Landlock required')
 _check(LIBC.prctl(38,1,0,0,0),'no_new_privs')
 handled=(1<<13)-1
 rules=Rules(handled);fd=_check(LIBC.syscall(444,ctypes.byref(rules),ctypes.sizeof(rules),0),'create ruleset')
 def add(path,rights):
  p=Path(path)
  if not p.exists():return
  handle=os.open(str(p.resolve()),os.O_PATH|os.O_CLOEXEC)
  try:
   if not p.is_dir():rights &= 7
   rule=Beneath(rights,handle);_check(LIBC.syscall(445,fd,1,ctypes.byref(rule),0),'add path '+str(p))
  finally:os.close(handle)
 ro=(1<<0)|(1<<2)|(1<<3)
 for path in ['/usr','/lib','/lib64','/bin',os.environ['SANDBOX_PYBASE'],os.environ['SANDBOX_EVALDEPS'],os.environ['SANDBOX_DEPS'],os.environ.get('SANDBOX_EVALPLUS_SOURCE',os.environ['SANDBOX_EVALDEPS']),sysconfig.get_path('stdlib'),sysconfig.get_path('platstdlib'),str(Path(os.__file__).resolve().parent)]:add(path,ro)
 for path in ['/etc/ld.so.cache','/etc/localtime','/dev/random','/dev/urandom']:add(path,1<<2)
 add('/dev/null',(1<<1)|(1<<2))
 add(os.environ['SANDBOX_TMP_ROOT'],handled & ~(1<<6) & ~(1<<7) & ~(1<<11))
 _check(LIBC.syscall(446,fd,0),'enforce Landlock');os.close(fd)
 sec=ctypes.CDLL(ctypes.util.find_library('seccomp'),use_errno=True)
 sec.seccomp_init.argtypes=[ctypes.c_uint32];sec.seccomp_init.restype=ctypes.c_void_p
 sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
 sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
 sec.seccomp_load.argtypes=[ctypes.c_void_p];sec.seccomp_release.argtypes=[ctypes.c_void_p]
 ctx=sec.seccomp_init(0x00050000|errno.EPERM)
 if not ctx:raise RuntimeError('seccomp_init failed')
 for name in ALLOWED:
  nr=sec.seccomp_syscall_resolve_name(name.encode())
  if nr>=0 and sec.seccomp_rule_add(ctx,0x7fff0000,nr,0)<0:raise RuntimeError('seccomp rule '+name)
 if sec.seccomp_load(ctx)<0:raise RuntimeError('seccomp_load failed')
 sec.seccomp_release(ctx)
 # These probes must succeed before executing any generated code.
 denied=Path(os.environ['SANDBOX_DENIED_SENTINEL'])
 try:
  with denied.open('rb') as f:f.read(1)
 except PermissionError:pass
 else:raise RuntimeError('host-file access was not denied')
 import socket
 try:socket.socket()
 except PermissionError:pass
 else:raise RuntimeError('socket creation was not denied')
 return {'landlock_abi':abi,'filesystem':'allowlisted runtime plus own temporary area','syscalls':'allowlist','new_sockets':False}
