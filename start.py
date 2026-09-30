#!/usr/bin/env python3
"""One-command launcher: makes a local .venv, installs dependencies, starts IVAP.
   python start.py                 normal start (auto-select CUDA / Intel XPU / CPU)
   python start.py --cuda          install NVIDIA CUDA PyTorch
   python start.py --xpu           install Intel XPU PyTorch
   python start.py --directml      install experimental Windows DirectML (AMD/DirectX 12)
   Add --desktop to any backend option to open the native Windows window.
   python start.py --bundle        download the common wheels into ./vendor"""
import os, sys, subprocess, venv
B = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(B, '.venv')
PY = os.path.join(V, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(V, 'bin', 'python')
OK = os.path.join(V, '.ivap_ready'); DESKTOP_OK = os.path.join(V, '.ivap_desktop_ready'); VD = os.path.join(B, 'vendor')
def run(*a, check=True): return subprocess.run(list(a), cwd=B).returncode if not check else subprocess.check_call(list(a), cwd=B)
if sys.version_info < (3, 9): sys.exit('Python 3.9+ is required')
if '--bundle' in sys.argv:
    for r in ('requirements.txt', 'requirements-desktop.txt'): run(sys.executable, '-m', 'pip', 'download', '-r', r, '-d', 'vendor', check=False)
    sys.exit('Wheels saved in ./vendor (specific to this OS + Python version)')
backend_flags = [x for x in ('--cuda', '--xpu', '--directml') if x in sys.argv]
if len(backend_flags) > 1: sys.exit('Choose only one GPU backend: --cuda, --xpu, or --directml')
backend = backend_flags[0] if backend_flags else ''
if backend == '--directml' and os.name != 'nt': sys.exit('DirectML setup is for Windows; use a ROCm PyTorch build for supported AMD GPUs on Linux.')
if not os.path.exists(PY): print('Creating local environment (.venv)...'); venv.create(V, with_pip=True)
if not os.path.exists(OK):
    off = ['--no-index', '--find-links', VD] if os.path.isdir(VD) and os.listdir(VD) else []
    run(PY, '-m', 'pip', 'install', '-r', 'requirements.txt', *off)
    open(OK, 'w').write('ok')
if backend:
    marker = os.path.join(V, '.ivap_' + backend[2:] + '_ready')
    if not os.path.exists(marker):
        if backend == '--cuda': run(PY, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu121')
        elif backend == '--xpu': run(PY, '-m', 'pip', 'install', '-r', 'requirements-intel-xpu.txt')
        elif backend == '--directml': run(PY, '-m', 'pip', 'install', '-r', 'requirements-amd-directml.txt')
        open(marker, 'w').write('ok')
    os.environ['IBVAP_DEVICE'] = {'--cuda':'cuda','--xpu':'xpu','--directml':'directml'}[backend]
else: os.environ.setdefault('IBVAP_DEVICE', 'auto')
if '--desktop' in sys.argv and not os.path.exists(DESKTOP_OK):
    off = ['--no-index', '--find-links', VD] if os.path.isdir(VD) and os.listdir(VD) else []
    run(PY, '-m', 'pip', 'install', '-r', 'requirements-desktop.txt', *off)
    open(DESKTOP_OK, 'w').write('ok')
target = 'desktop_app.py' if '--desktop' in sys.argv else 'ivap.py'
sys.exit(subprocess.call([PY, target], cwd=B))
